"""Shared helpers for reading VSE strip frames as numpy RGB arrays."""

from __future__ import annotations

import os

import bpy


def active_sequence_strip(context):
    """Return the active MOVIE or IMAGE strip in the VSE, or None."""
    scene = context.scene
    seq_editor = scene.sequence_editor
    if not seq_editor:
        return None
    strip = seq_editor.active_strip
    if strip and strip.type in {"MOVIE", "IMAGE"}:
        return strip
    # Fall back to the first movie/image strip covering the current frame.
    all_strips = getattr(seq_editor, "strips_all", None) or seq_editor.sequences_all
    for s in all_strips:
        if s.type in {"MOVIE", "IMAGE"} and s.frame_final_start <= scene.frame_current <= s.frame_final_end:
            return s
    return None


class FrameReader:
    """Reads frames of a VSE strip by scene-frame number, as HxWx3 uint8 RGB."""

    def __init__(self, strip):
        self.strip = strip
        self.type = strip.type
        self._cap = None
        self._frame_count = 0

        if self.type == "MOVIE":
            import cv2

            path = bpy.path.abspath(strip.filepath)
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise IOError(f"Could not open movie: {path}")
            self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        elif self.type == "IMAGE":
            self._dir = bpy.path.abspath(strip.directory)
            self._names = [e.filename for e in strip.elements]
            self._frame_count = len(self._names)

    def source_index(self, scene_frame: int) -> int:
        """Map a scene frame to the strip's source frame index."""
        idx = int(scene_frame - self.strip.frame_start)
        return max(0, min(idx, self._frame_count - 1))

    def get_frame(self, scene_frame: int):
        import cv2
        import numpy as np

        src = self.source_index(scene_frame)

        if self.type == "MOVIE":
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, src)
            ok, bgr = self._cap.read()
            if not ok:
                raise IOError(f"Failed to read movie frame {src}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # IMAGE sequence
        path = os.path.join(self._dir, self._names[src])
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Failed to read image frame: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def numpy_to_image(name: str, rgb, mask=None):
    """Create/update a bpy.data.images image from an HxWx3 uint8 RGB array.

    If `mask` (HxW bool) is given, matched pixels are tinted so the user sees
    the current selection as a live overlay. Returns the bpy image.
    """
    import numpy as np

    h, w = rgb.shape[:2]
    rgba = np.empty((h, w, 4), dtype=np.float32)
    rgba[..., :3] = rgb.astype(np.float32) / 255.0
    rgba[..., 3] = 1.0

    if mask is not None:
        # Tint the selected region toward cyan for visibility.
        sel = mask.astype(bool)
        rgba[sel, 0] *= 0.4
        rgba[sel, 1] = rgba[sel, 1] * 0.6 + 0.4
        rgba[sel, 2] = rgba[sel, 2] * 0.6 + 0.4

    img = bpy.data.images.get(name)
    if img is None or tuple(img.size) != (w, h):
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(name, width=w, height=h, alpha=True, float_buffer=True)

    # Blender image rows are bottom-up; our arrays are top-down.
    flipped = np.flipud(rgba)
    img.pixels.foreach_set(flipped.ravel())
    img.update()
    return img


def show_image_in_editor(context, image):
    """Point the first IMAGE_EDITOR area at `image` so the user can pick on it."""
    for area in context.screen.areas:
        if area.type == "IMAGE_EDITOR":
            area.spaces.active.image = image
            return area
    return None
