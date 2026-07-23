"""Matte writing and result-strip creation, shared by pick and track."""

from __future__ import annotations

import os

import numpy as np

RESULT_STRIP_NAME = "AUTO_Matte"


def _all_strips(se):
    # Blender 5.x renamed Sequence -> Strip (sequences_all -> strips_all).
    return getattr(se, "strips_all", None) or se.sequences_all


def _strip_collection(se):
    return getattr(se, "strips", None) or se.sequences


def place_on_canvas(alpha_src: np.ndarray, placement) -> np.ndarray:
    """Composite a source-resolution alpha onto a render-resolution canvas.

    placement = (render_w, render_h, off_x, off_y, scale_x, scale_y), matching
    the strip's transform in the preview (offset in render pixels from the frame
    centre, scale on the strip's native size). This makes the matte match the
    project resolution and sit exactly where the footage appears.
    """
    import cv2

    rw, rh, ox, oy, sx, sy = placement
    rw, rh = int(round(rw)), int(round(rh))
    h, w = alpha_src.shape
    dw = max(1, int(round(w * sx)))
    dh = max(1, int(round(h * sy)))
    resized = cv2.resize(alpha_src, (dw, dh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((rh, rw), dtype=np.uint8)
    cx = rw / 2.0 + ox           # placed-image centre (render px, x right)
    cy = rh / 2.0 - oy           # y down
    x0 = int(round(cx - dw / 2.0))
    y0 = int(round(cy - dh / 2.0))

    sx0, sy0 = max(0, -x0), max(0, -y0)          # crop offset in the resized img
    dx0, dy0 = max(0, x0), max(0, y0)            # paste offset in the canvas
    x1, y1 = min(rw, x0 + dw), min(rh, y0 + dh)
    if x1 > dx0 and y1 > dy0:
        canvas[dy0:y1, dx0:x1] = resized[sy0:sy0 + (y1 - dy0), sx0:sx0 + (x1 - dx0)]
    return canvas


def write_matte(out_dir: str, frame: int, alpha_u8: np.ndarray, placement=None) -> str:
    """Write an RGBA PNG matte (white fill, given 0..255 alpha). Returns path.

    If `placement` is given, the alpha is composited onto a render-resolution
    canvas first, so the matte matches the project resolution.
    """
    import cv2

    if placement is not None:
        alpha_u8 = place_on_canvas(alpha_u8, placement)
    h, w = alpha_u8.shape
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[..., :3] = 255
    bgra[..., 3] = alpha_u8
    path = os.path.join(out_dir, f"matte_{frame:05d}.png")
    cv2.imwrite(path, bgra)
    return path


def rebuild_result_strip(context, out_dir: str, frames) -> None:
    """(Re)create a single image strip spanning the written matte frames.

    Any previous result strip is removed first, so pick and track keep updating
    the same strip instead of piling up. Frames are contiguous in practice
    (tracking fills outward from the anchor), so they map 1:1 to strip elements.
    """
    frames = sorted(frames)
    if not frames:
        return

    scene = context.scene
    se = scene.sequence_editor or scene.sequence_editor_create()

    coll = _strip_collection(se)
    for s in [s for s in _all_strips(se) if s.name == RESULT_STRIP_NAME]:
        coll.remove(s)

    first = os.path.join(out_dir, f"matte_{frames[0]:05d}.png")
    channel = max((s.channel for s in _all_strips(se)), default=0) + 1
    strip = coll.new_image(
        name=RESULT_STRIP_NAME,
        filepath=first,
        channel=channel,
        frame_start=frames[0],
    )
    for f in frames[1:]:
        strip.elements.append(f"matte_{f:05d}.png")
    # Composite over the footage by default.
    try:
        strip.blend_type = "ALPHA_OVER"
    except (AttributeError, TypeError):
        pass
