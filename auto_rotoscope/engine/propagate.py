"""Frame-to-frame mask propagation using OpenCV optical flow.

SAM 2.1's native video memory is not exportable to ONNX, so we approximate
temporal tracking: dense optical flow moves the point prompts (and warps the
previous low-res mask) from frame N to N+1, then SAM re-segments frame N+1.
This gives automatic tracking that stays anchored to the object while remaining
fully bundlable (OpenCV only; no PyTorch).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .sam2 import SAM2Image, LOW_RES


def _to_gray(rgb: np.ndarray):
    import cv2

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _dense_flow(prev_gray, cur_gray, quality: str = "BALANCED"):
    import cv2

    # Fewer levels / iterations = faster, coarser tracking.
    presets = {
        "FAST": dict(pyr_scale=0.5, levels=2, winsize=13, iterations=2, poly_n=5, poly_sigma=1.1),
        "BALANCED": dict(pyr_scale=0.5, levels=3, winsize=21, iterations=3, poly_n=5, poly_sigma=1.1),
        "ACCURATE": dict(pyr_scale=0.5, levels=4, winsize=31, iterations=5, poly_n=7, poly_sigma=1.5),
    }
    p = presets.get(quality, presets["BALANCED"])
    return cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, flags=0, **p)


def _advect_points(points_xy: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Move each point by the flow vector sampled at its (clamped) location."""
    h, w = flow.shape[:2]
    moved = points_xy.copy().astype(np.float32)
    for i, (x, y) in enumerate(points_xy):
        xi = int(min(max(round(x), 0), w - 1))
        yi = int(min(max(round(y), 0), h - 1))
        dx, dy = flow[yi, xi]
        moved[i, 0] = min(max(x + float(dx), 0.0), w - 1)
        moved[i, 1] = min(max(y + float(dy), 0.0), h - 1)
    return moved


def track(
    sam: SAM2Image,
    get_frame: Callable[[int], np.ndarray],
    anchor_index: int,
    frame_start: int,
    frame_end: int,
    points_xy: np.ndarray,
    labels: np.ndarray,
    on_mask: Callable[[int, np.ndarray], None],
    quality: str = "BALANCED",
    progress: Callable[[int, int], bool] | None = None,
) -> int:
    """Segment the anchor frame, then propagate across [frame_start, frame_end].

    get_frame(i) -> HxWx3 uint8 RGB.
    on_mask(i, mask_bool) is called for every frame with the produced matte.
    progress(done, total) -> return False to cancel.

    Returns the number of frames written.
    """
    # 1) Segment the anchor frame from the user's prompts.
    anchor_rgb = get_frame(anchor_index)
    emb = sam.encode(anchor_rgb)
    mask_bool, low_res, _ = sam.decode(emb, points_xy, labels)
    on_mask(anchor_index, mask_bool)

    total = frame_end - frame_start + 1
    written = 1

    # Helper to walk outward from the anchor in one direction.
    def _walk(indices: list[int]):
        nonlocal written
        prev_gray = _to_gray(anchor_rgb)
        cur_points = points_xy.copy().astype(np.float32)
        cur_low = low_res.copy()

        for idx in indices:
            cur_rgb = get_frame(idx)
            cur_gray = _to_gray(cur_rgb)

            flow = _dense_flow(prev_gray, cur_gray, quality)
            cur_points = _advect_points(cur_points, flow)

            emb_i = sam.encode(cur_rgb)
            mask_i, cur_low, iou = sam.decode(emb_i, cur_points, labels, mask_input=cur_low)
            on_mask(idx, mask_i)

            prev_gray = cur_gray
            written += 1
            if progress and not progress(written, total):
                return False
        return True

    # Forward pass: anchor+1 .. end
    forward = list(range(anchor_index + 1, frame_end + 1))
    if not _walk(forward):
        return written

    # Backward pass: anchor-1 .. start
    backward = list(range(anchor_index - 1, frame_start - 1, -1))
    _walk(backward)

    return written
