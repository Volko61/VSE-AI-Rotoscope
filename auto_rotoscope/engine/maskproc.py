"""Mask post-processing shared by the preview overlay and the matte export.

- refine_mask: optionally reduce the mask to a single closed shape (largest
  connected component with its holes filled), removing stray islands.
- feather_alpha: turn a boolean mask into a soft-edged 0..255 alpha.
- mask_to_contours: outline polygons for the preview overlay.
"""

from __future__ import annotations

import numpy as np


def refine_mask(mask_bool: np.ndarray, single_shape: bool) -> np.ndarray:
    """Return a cleaned boolean mask.

    When single_shape is True, keep only the largest connected component and
    fill its interior holes so the result is one closed shape.
    """
    if not single_shape:
        return mask_bool

    import cv2

    m = mask_bool.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 2:  # background + at most one component
        keep = mask_bool
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        keep = labels == largest

    # Fill holes: draw the external contour(s) filled.
    filled = np.zeros(keep.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(
        keep.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled > 0


def feather_alpha(mask_bool: np.ndarray, feather_px: float) -> np.ndarray:
    """Boolean mask -> uint8 alpha (0..255), with soft edges if feather_px > 0."""
    alpha = (mask_bool.astype(np.uint8)) * 255
    if feather_px and feather_px > 0:
        import cv2

        k = int(round(feather_px)) * 2 + 1  # odd kernel
        sigma = max(feather_px / 2.0, 0.8)
        alpha = cv2.GaussianBlur(alpha, (k, k), sigma)
    return alpha


def mask_to_contours(mask_bool: np.ndarray):
    """External contour polygons as a list of (N,2) float arrays (source pixels)."""
    import cv2

    m = (mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).astype(np.float32) for c in contours if len(c) >= 2]
