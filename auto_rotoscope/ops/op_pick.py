"""Interactive point picking directly in the Sequencer preview.

No separate Image Editor is needed: the operator runs modally over the VSE
preview region and captures clicks. Visual feedback (markers + mask outline) is
drawn by the shared persistent overlay in ui/overlay.py.
"""

from __future__ import annotations

import bpy
import numpy as np

from . import common
from . import output
from ..engine import loader
from ..engine import maskproc
from ..preferences import get_prefs, resolve_output_dir
from ..properties import MODELS
from ..ui import overlay

# Kept for backward-compat with op_clear (no preview image is created anymore).
PREVIEW_NAME = "AUTO_Preview"


def _points_to_pixels(props, w, h):
    """Convert stored normalized (u, v, bottom-left origin) points to pixels."""
    if len(props.points) == 0:
        return None, None
    coords = np.zeros((len(props.points), 2), dtype=np.float32)
    labels = np.zeros(len(props.points), dtype=np.int32)
    for i, p in enumerate(props.points):
        coords[i, 0] = p.x * w
        coords[i, 1] = (1.0 - p.y) * h  # flip: image origin is bottom-left
        labels[i] = 1 if p.positive else 0
    return coords, labels


def _find_preview_region(context):
    """Return (area, region) for the Sequencer area actually showing the preview.

    Every SEQUENCE_EDITOR area carries a PREVIEW region even in timeline mode,
    but it is collapsed to 1x1 there — so pick the largest real one.
    """
    best = (None, None)
    best_area = 0
    for area in context.window.screen.areas:
        if area.type != "SEQUENCE_EDITOR":
            continue
        view = area.spaces.active.view_type
        if view not in {"PREVIEW", "SEQUENCER_PREVIEW"}:
            continue
        for region in area.regions:
            if region.type == "PREVIEW" and region.width > 1 and region.height > 1:
                size = region.width * region.height
                if size > best_area:
                    best_area = size
                    best = (area, region)
    return best


class SAM2_OT_pick(bpy.types.Operator):
    bl_idname = "auto_roto.pick"
    bl_label = "Pick Object (SAM2)"
    bl_description = (
        "Click the object in the video preview to add points. "
        "Left click adds, Ctrl+Left click removes. Enter to confirm, Esc to cancel"
    )
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        props = context.scene.auto_roto

        strip = common.active_sequence_strip(context)
        if strip is None:
            self.report({"ERROR"}, "Select a Movie or Image strip in the VSE first")
            return {"CANCELLED"}

        self._area, self._region = _find_preview_region(context)
        if self._region is None:
            self.report({"ERROR"}, "Show the Sequencer preview to pick points")
            return {"CANCELLED"}

        try:
            self._reader = common.FrameReader(strip)
            self._rgb = self._reader.get_frame(context.scene.frame_current)
        except Exception as exc:
            self.report({"ERROR"}, f"Could not read frame: {exc}")
            return {"CANCELLED"}

        self._h, self._w = self._rgb.shape[:2]
        self._anchor = context.scene.frame_current
        self._low = None
        self._last_mask = None

        # Strip placement in the render frame (offset in render pixels from the
        # centre, scale on native size). The preview view space is render pixels
        # centred at the frame centre, so clicks map straight through it.
        tr = getattr(strip, "transform", None)
        self._ox = getattr(tr, "offset_x", 0.0)
        self._oy = getattr(tr, "offset_y", 0.0)
        self._sx = getattr(tr, "scale_x", 1.0) or 1.0
        self._sy = getattr(tr, "scale_y", 1.0) or 1.0
        rnd = context.scene.render
        self._rw, self._rh = rnd.resolution_x, rnd.resolution_y
        overlay.set_geometry(self._ox, self._oy, self._sx, self._sy, self._w, self._h)

        self._refresh_mask(context)
        context.window_manager.modal_handler_add(self)
        props.status = "Picking… Left=add, Ctrl+Left=remove, Enter=confirm"
        overlay.tag_redraw_previews(context)
        return {"RUNNING_MODAL"}

    # -- modal loop ---------------------------------------------------------

    def modal(self, context, event):
        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            return self._finish(context, confirm=True)
        if event.type == "ESC" and event.value == "PRESS":
            return self._finish(context, confirm=False)

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            region = self._region
            rx = event.mouse_x - region.x
            ry = event.mouse_y - region.y
            if not (0 <= rx <= region.width and 0 <= ry <= region.height):
                return {"PASS_THROUGH"}  # click outside the preview region

            vx, vy = region.view2d.region_to_view(rx, ry)
            nx = (vx - self._ox) / (self._w * self._sx) + 0.5
            ny = 0.5 - (vy - self._oy) / (self._h * self._sy)  # top-origin
            u = nx
            v = 1.0 - ny  # bottom-origin, matching _points_to_pixels

            props = context.scene.auto_roto
            pt = props.points.add()
            pt.x = min(max(float(u), 0.0), 1.0)
            pt.y = min(max(float(v), 0.0), 1.0)
            pt.positive = not event.ctrl
            props.status = f"{len(props.points)} point(s)"
            self._refresh_mask(context)
            overlay.tag_redraw_previews(context)
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    # -- mask ---------------------------------------------------------------

    def _refresh_mask(self, context):
        props = context.scene.auto_roto
        coords, labels = _points_to_pixels(props, self._w, self._h)
        if coords is None:
            self._last_mask = None
            overlay.set_contours([])
            return
        try:
            prefs = get_prefs(context)
            sam = loader.load(MODELS[props.model]["dir"], prefs.provider_mode)
            emb = sam.encode(self._rgb)
            mask, self._low, _ = sam.decode(emb, coords, labels)
            props.active_provider = loader.active_provider()
            self._last_mask = maskproc.refine_mask(mask, props.single_shape)
            overlay.set_contours(maskproc.mask_to_contours(self._last_mask))
        except Exception as exc:
            props.status = f"Preview error: {exc}"

    # -- teardown -----------------------------------------------------------

    def _finish(self, context, confirm):
        props = context.scene.auto_roto
        self._reader.release()
        overlay.tag_redraw_previews(context)
        if not confirm:
            props.status = "Picking cancelled"
            return {"CANCELLED"}

        props.anchor_frame = self._anchor

        # Bake the anchor frame's matte and add it as a strip right away.
        if self._last_mask is not None:
            try:
                out_dir = resolve_output_dir(context)
                alpha = maskproc.feather_alpha(self._last_mask, props.feather)
                placement = (self._rw, self._rh, self._ox, self._oy, self._sx, self._sy)
                output.write_matte(out_dir, self._anchor, alpha, placement)
                output.rebuild_result_strip(context, out_dir, [self._anchor])
                props.status = f"Matte added at frame {self._anchor}. Track to extend it."
            except Exception as exc:
                props.status = f"Anchor set, but matte failed: {exc}"
        else:
            props.status = f"Anchor set with {len(props.points)} point(s). Ready to Track."
        return {"FINISHED"}


def register():
    bpy.utils.register_class(SAM2_OT_pick)


def unregister():
    bpy.utils.unregister_class(SAM2_OT_pick)
