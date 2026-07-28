#!/usr/bin/env python3
"""Generate the listing icon for extensions.blender.org.

The design says "rotoscoping" in one glance, at any size: a bold subject
silhouette cleanly cut out of an alpha checkerboard, traced by a dashed
selection outline, with the pick marker on it.

Only numpy is needed (Blender bundles it), so this runs either standalone or
through Blender's interpreter:

    python scripts/make_icon.py
    blender -b --python scripts/make_icon.py

Writes PNGs to ./branding/.
"""

from __future__ import annotations

import os
import struct
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "branding")

SIZES = (1024, 512, 256, 128)
SS = 4  # supersampling factor, for anti-aliasing

# Palette. Orange is Blender's accent, so the icon sits naturally on the site.
ORANGE_HI = np.array([0xF5, 0x9E, 0x1B], dtype=np.float32)
ORANGE_LO = np.array([0xD9, 0x66, 0x08], dtype=np.float32)
CHECK_A = np.array([0x33, 0x33, 0x36], dtype=np.float32)
CHECK_B = np.array([0x28, 0x28, 0x2B], dtype=np.float32)
OUTLINE = np.array([0xFF, 0xFF, 0xFF], dtype=np.float32)


def write_png(path: str, rgb: np.ndarray) -> None:
    """Write an 8-bit RGB PNG without any imaging library."""
    h, w, _ = rgb.shape
    raw = b"".join(
        b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as fh:
        fh.write(png)


def _ellipse(xx, yy, cx, cy, rx, ry):
    """Signed field: < 1 inside the ellipse."""
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2


def _superellipse(xx, yy, cx, cy, rx, ry, p):
    """Signed field: < 1 inside. p > 2 flattens the top and steepens the sides.

    A plain ellipse tapers to a point, which is what makes a head-over-dome
    silhouette read as a chess pawn. p ~ 3 reaches near-full width just below
    the neck, the way real shoulders do.
    """
    return np.abs((xx - cx) / rx) ** p + np.abs((yy - cy) / ry) ** p


def subject_mask(n: int) -> np.ndarray:
    """Boolean head-and-shoulders silhouette on an n x n grid (0..1 coords).

    Sized to leave a clear margin on every side: the dashed outline grows
    outward from this, and both need to stay inside the tile.
    """
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    xx, yy = (x + 0.5) / n, (y + 0.5) / n

    head = _ellipse(xx, yy, 0.50, 0.275, 0.125, 0.145) < 1.0
    # Shoulders: ~2.6 head-widths across, squared off so they broaden fast
    # under the neck. Cropped flat above the lower margin.
    shoulders = (
        (_superellipse(xx, yy, 0.50, 0.88, 0.325, 0.360, 3.0) < 1.0)
        & (yy < 0.855)
    )
    # Short, narrow neck: just enough to bridge head and shoulders.
    neck = (np.abs(xx - 0.50) < 0.046) & (yy > 0.38) & (yy < 0.56)
    return head | shoulders | neck


def _grow(mask: np.ndarray, steps: int) -> np.ndarray:
    """Morphological dilation by shifting, without wrapping at the borders.

    np.roll is circular, so dilating near an edge would smear the silhouette
    onto the opposite side of the tile. Padding with False prevents that.
    """
    m = np.pad(mask, steps, constant_values=False)
    for _ in range(steps):
        m = (
            m
            | np.roll(m, 1, 0) | np.roll(m, -1, 0)
            | np.roll(m, 1, 1) | np.roll(m, -1, 1)
        )
    return m[steps:-steps, steps:-steps]


def dashed_ring(mask: np.ndarray, n: int) -> np.ndarray:
    """Marching-ants band hugging the outside of `mask`."""
    band_px = max(2, int(round(n * 0.012)))
    gap_px = max(1, int(round(n * 0.008)))
    ring = _grow(mask, band_px + gap_px) & ~_grow(mask, gap_px)

    # Diagonal dashes, so the band breaks up like a selection marquee.
    y, x = np.mgrid[0:n, 0:n]
    period = max(6, int(round(n * 0.045)))
    dash = ((x + y) % period) < (period * 0.55)
    return ring & dash


def pick_marker(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The '+' pick point: (glyph mask, dark halo mask).

    Small and set on the upper chest. Big enough to read at 128px, small
    enough not to look like a medical cross.
    """
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    xx, yy = (x + 0.5) / n, (y + 0.5) / n
    cx, cy = 0.50, 0.715
    arm, thick = 0.034, 0.0105
    horiz = (np.abs(xx - cx) < arm) & (np.abs(yy - cy) < thick)
    vert = (np.abs(yy - cy) < arm) & (np.abs(xx - cx) < thick)
    glyph = horiz | vert
    halo = _ellipse(xx, yy, cx, cy, arm * 1.5, arm * 1.5) < 1.0
    return glyph, halo


def render(size: int) -> np.ndarray:
    n = size * SS
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    yy = (y + 0.5) / n

    # Alpha checkerboard background - the universal "this is a matte" cue.
    cell = max(1, int(round(n / 16)))
    checker = (((x // cell).astype(int) + (y // cell).astype(int)) % 2) == 0
    img = np.where(checker[..., None], CHECK_A, CHECK_B).astype(np.float32)

    # Subject, filled with a soft vertical orange gradient.
    mask = subject_mask(n)
    t = np.clip((yy - 0.15) / 0.85, 0.0, 1.0)[..., None]
    orange = ORANGE_HI * (1.0 - t) + ORANGE_LO * t
    img = np.where(mask[..., None], orange, img)

    # Dashed selection outline.
    img = np.where(dashed_ring(mask, n)[..., None], OUTLINE, img)

    # Pick marker, on a darkened halo so it stays legible over the orange.
    glyph, halo = pick_marker(n)
    img = np.where((halo & ~glyph)[..., None], img * 0.45, img)
    img = np.where(glyph[..., None], OUTLINE, img)

    # Box-filter down to the target size: that is the anti-aliasing.
    img = img.reshape(size, SS, size, SS, 3).mean(axis=(1, 3))
    return np.clip(img, 0, 255)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    for size in SIZES:
        path = os.path.join(OUT_DIR, f"icon_{size}.png")
        write_png(path, render(size))
        print(f"  {os.path.getsize(path):>7,} B  {os.path.relpath(path, ROOT)}")
    print(f"\nIcons written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
