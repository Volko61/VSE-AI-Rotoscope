#!/usr/bin/env python3
"""End-to-end smoke test for the *installed* extension, run inside Blender.

Verifies what an extensions.blender.org reviewer checks first: the add-on
registers, its dependencies import, the bundled model actually runs inference,
and the operators/panel are present. It runs headless, on a synthetic image,
and needs no .blend file or footage.

Usage (after `blender --command extension install-file -r user_default -e <zip>`):

    blender -b --python scripts/smoke_test.py

Exits non-zero on the first failure, so it works as a release gate.
"""

import sys
import traceback

import bpy

PKG = "bl_ext.user_default.auto_rotoscope"

_failures = []


def check(label):
    """Decorator: run a check, record pass/fail, never abort the whole run."""
    def wrap(fn):
        try:
            detail = fn()
        except Exception:
            _failures.append(label)
            print(f"FAIL  {label}")
            traceback.print_exc()
        else:
            print(f"ok    {label}" + (f"  ({detail})" if detail else ""))
        return fn
    return wrap


@check("extension is installed and enabled")
def _enabled():
    if PKG not in bpy.context.preferences.addons:
        raise AssertionError(f"{PKG} not enabled. Install the built ZIP first.")
    return PKG


@check("bundled wheels import")
def _wheels():
    import cv2
    import onnxruntime as ort
    import numpy
    return f"onnxruntime {ort.__version__}, cv2 {cv2.__version__}, numpy {numpy.__version__}"


@check("execution providers resolve")
def _providers():
    mod = sys.modules[f"{PKG}.engine.ort_session"]
    assert mod.is_available(), mod.import_error()
    return ", ".join(mod.available_providers())


@check("operators and panel are registered")
def _registered():
    for op in ("pick", "track", "clear", "remove_point"):
        assert hasattr(bpy.ops.auto_roto, op), f"missing operator auto_roto.{op}"
    assert hasattr(bpy.types, "SAM2_PT_panel"), "sidebar panel not registered"
    assert hasattr(bpy.types.Scene, "auto_roto"), "scene property group missing"
    return "4 operators + panel"


@check("no network access is attempted")
def _offline():
    # The manifest requests no `network` permission, so nothing may open a
    # socket. Poison socket creation and exercise a model load.
    import socket

    orig = socket.socket

    def blocked(*a, **kw):
        raise AssertionError("the add-on opened a socket - it must stay offline")

    socket.socket = blocked
    try:
        loader = sys.modules[f"{PKG}.engine.loader"]
        loader.load("sam2.1_hiera_tiny", "CPU")
    finally:
        socket.socket = orig
    return "socket() stayed unused during model load"


@check("SAM 2.1 inference produces a mask")
def _inference():
    import numpy as np

    loader = sys.modules[f"{PKG}.engine.loader"]
    sam = loader.load("sam2.1_hiera_tiny", "CPU")

    # Synthetic frame: a bright disc on a dark background.
    h, w = 360, 640
    rgb = np.full((h, w, 3), 30, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 < 90**2
    rgb[disc] = (230, 190, 60)

    emb = sam.encode(rgb)
    coords = np.array([[w / 2, h / 2]], dtype=np.float32)   # one positive point
    labels = np.array([1], dtype=np.float32)
    mask, low, iou = sam.decode(emb, coords, labels)

    assert mask.shape == (h, w), f"mask shape {mask.shape} != frame {(h, w)}"
    assert low.shape == (1, 1, 256, 256), f"low-res logits shape {low.shape}"
    covered = mask.sum()
    assert covered > 0, "empty mask - the decoder segmented nothing"
    # The prompt is the disc centre, so the mask should land on the disc, not
    # cover the whole frame. SAM habitually adds a margin around hard edges, so
    # this is a sanity bound, not an accuracy measurement.
    overlap = (mask & disc).sum() / max(1, covered)
    assert overlap > 0.7, f"mask only {overlap:.0%} inside the prompted disc"
    return f"{covered} px, {overlap:.0%} on target, IoU {iou:.2f}"


@check("mask post-processing and matte export")
def _export():
    import os
    import tempfile

    import numpy as np

    maskproc = sys.modules[f"{PKG}.engine.maskproc"]
    output = sys.modules[f"{PKG}.ops.output"]

    mask = np.zeros((120, 200), dtype=bool)
    mask[30:90, 50:150] = True
    mask[50:60, 70:80] = False          # a hole, to exercise single_shape
    mask[5:8, 5:8] = True               # a stray island

    refined = maskproc.refine_mask(mask, single_shape=True)
    assert not refined[5:8, 5:8].any(), "stray island survived single_shape"
    assert refined[50:60, 70:80].all(), "hole was not filled"

    alpha = maskproc.feather_alpha(refined, 3.0)
    assert alpha.dtype == np.uint8 and alpha.max() == 255

    contours = maskproc.mask_to_contours(refined)
    assert contours, "no contours produced for the overlay"

    with tempfile.TemporaryDirectory() as d:
        path = output.write_matte(d, 7, alpha, placement=(400, 300, 0, 0, 1.0, 1.0))
        assert os.path.basename(path) == "matte_00007.png", path
        assert os.path.getsize(path) > 0, "empty PNG written"
    return f"{len(contours)} contour(s), RGBA PNG at render resolution"


print("\n" + "=" * 60)
if _failures:
    print(f"{len(_failures)} check(s) FAILED: {', '.join(_failures)}")
    sys.exit(1)
print("All checks passed.")
sys.exit(0)
