#!/usr/bin/env python3
"""Fetch build assets for the Auto Rotoscope extension: model + platform wheels.

These binaries are NOT committed to the repo (too large). Run this once before
building the extension. It:

  * downloads the SAM 2.1 tiny ONNX bundle (a .zip), extracts it, and writes
    auto_rotoscope/models/sam2.1_hiera_tiny/{encoder,decoder}.onnx
  * downloads onnxruntime(-directml) + opencv-python-headless wheels for the
    target Python (cp313 by default) into auto_rotoscope/wheels/
  * rewrites the `wheels = [...]` block in blender_manifest.toml with the exact
    filenames actually downloaded (no manual copy-paste needed).

Requirements: pip, internet access. The download works from any local Python;
--python-version selects the *target* interpreter for the wheels.

Usage:
    python scripts/fetch_assets.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "auto_rotoscope")
WHEELS_DIR = os.path.join(PKG, "wheels")
MODEL_DIR = os.path.join(PKG, "models", "sam2.1_hiera_tiny")
MANIFEST = os.path.join(PKG, "blender_manifest.toml")

# Blender 5.1+ ships Python 3.13 (Blender 4.2–5.0 used 3.11). We target 3.13 —
# bundling both would exceed the 200 MB ZIP cap. Change PY_TAG to "311" (and
# blender_version_min back to 4.2.0) if you instead target Blender 4.2–5.0.
PY_TAG = "313"

# onnxruntime is left unpinned so pip picks the newest wheel with a cp313 build
# (directml >= 1.24.4 has it). OpenCV is abi3, so one wheel covers all versions.
OPENCV_VERSION = "4.10.0.84"

# SAM 2.1 tiny ONNX bundle (Apache-2.0), packaged as a .zip by the samexporter
# author. If the filename 404s, open the repo and update MODEL_ZIP_URL:
#   https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models
MODEL_ZIP_URL = (
    "https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models/"
    "resolve/main/sam2.1_hiera_tiny_20260221.zip"
)

# (package spec, platform tag) pairs to download as wheels. The tags map to
# Blender's `platforms` identifiers as follows (see blender_ext.py):
#   win_amd64              -> windows-x64
#   manylinux*_x86_64      -> linux-x64
#   macosx_*_arm64         -> macos-arm64  (the version prefix is ignored)
#
# macOS Intel is not built: Blender dropped macOS x86_64 builds at 5.0 and we
# require 5.1+. Apple Silicon only.
#
# onnxruntime is pinned to 1.23.2 on macOS: it is the oldest release with
# arch-specific mac wheels (16 MB) instead of universal2 (34 MB, half of it a
# dead x86_64 slice), and the saving is what keeps the macOS ZIP off the cap.
# It requires macOS 13+. Newer releases only publish macosx_14_0_arm64, which
# would raise the OS floor to macOS 14 for no size benefit.
ORT_MACOS_VERSION = "1.23.2"

WHEEL_TARGETS = [
    ("onnxruntime-directml", "win_amd64"),
    ("onnxruntime", "manylinux_2_28_x86_64"),
    (f"onnxruntime=={ORT_MACOS_VERSION}", "macosx_13_0_arm64"),
    (f"opencv-python-headless=={OPENCV_VERSION}", "win_amd64"),
    (f"opencv-python-headless=={OPENCV_VERSION}", "manylinux2014_x86_64"),
    (f"opencv-python-headless=={OPENCV_VERSION}", "macosx_11_0_arm64"),
]


def _download(url: str, dest: str):
    print(f"  → {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            read += len(chunk)
            if total:
                pct = read * 100 // total
                print(f"    {read >> 20} / {total >> 20} MiB ({pct}%)", end="\r")
    os.replace(tmp, dest)
    print(f"    saved {os.path.getsize(dest) >> 20} MiB          ")


def fetch_models():
    print("Fetching SAM 2.1 tiny ONNX model…")
    os.makedirs(MODEL_DIR, exist_ok=True)
    enc_dst = os.path.join(MODEL_DIR, "encoder.onnx")
    dec_dst = os.path.join(MODEL_DIR, "decoder.onnx")
    if os.path.isfile(enc_dst) and os.path.isfile(dec_dst):
        print("  encoder.onnx + decoder.onnx already present, skipping")
        return

    with tempfile.TemporaryDirectory() as tmp:
        zpath = os.path.join(tmp, "model.zip")
        try:
            _download(MODEL_ZIP_URL, zpath)
        except Exception as exc:
            print(f"  !! Could not download the model zip: {exc}")
            print(f"     Download it manually from the repo and extract encoder/decoder")
            print(f"     .onnx to {MODEL_DIR}\\{{encoder,decoder}}.onnx")
            return

        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(tmp)

        # Find the encoder/decoder .onnx anywhere in the extracted tree.
        # Prefer plain .onnx over optimized .ort variants.
        enc = dec = None
        for root, _dirs, files in os.walk(tmp):
            for f in files:
                low = f.lower()
                if not low.endswith(".onnx"):
                    continue
                full = os.path.join(root, f)
                if "encoder" in low and enc is None:
                    enc = full
                elif "decoder" in low and dec is None:
                    dec = full

        if not enc or not dec:
            print("  !! Could not locate encoder/decoder .onnx inside the zip.")
            print(f"     Extracted files: place them manually in {MODEL_DIR}")
            return

        shutil.copyfile(enc, enc_dst)
        shutil.copyfile(dec, dec_dst)
        print(f"  encoder.onnx  ({os.path.getsize(enc_dst) >> 20} MiB)")
        print(f"  decoder.onnx  ({os.path.getsize(dec_dst) >> 20} MiB)")


def fetch_wheels():
    print("Downloading platform wheels…")
    if os.path.isdir(WHEELS_DIR):
        for f in os.listdir(WHEELS_DIR):
            if f.endswith(".whl"):
                os.remove(os.path.join(WHEELS_DIR, f))  # drop stale wheels
    os.makedirs(WHEELS_DIR, exist_ok=True)
    for spec, platform in WHEEL_TARGETS:
        print(f"  {spec}  [{platform}]")
        cmd = [
            sys.executable, "-m", "pip", "download", spec,
            "--only-binary=:all:", "--no-deps",
            "--python-version", PY_TAG,
            "--implementation", "cp",
            "--platform", platform,
            "--dest", WHEELS_DIR,
        ]
        subprocess.run(cmd, check=True)


def update_manifest():
    wheels = sorted(f for f in os.listdir(WHEELS_DIR) if f.endswith(".whl"))
    block = "wheels = [\n" + "".join(f'  "./wheels/{w}",\n' for w in wheels) + "]"

    with open(MANIFEST, "r", encoding="utf-8") as fh:
        text = fh.read()

    new_text, n = re.subn(r"wheels = \[.*?\]", block, text, count=1, flags=re.DOTALL)
    if n == 1 and new_text != text:
        with open(MANIFEST, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        print("\nUpdated wheels[] in blender_manifest.toml:")
    else:
        print("\n(Manifest unchanged — paste this block manually if needed:)")
    for w in wheels:
        print(f'  ./wheels/{w}')


if __name__ == "__main__":
    fetch_models()
    fetch_wheels()
    update_manifest()
    print("\nDone. Next: blender --command extension build --split-platforms")
