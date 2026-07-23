"""Interactive tracking: watch the mask propagate, pause, step, go back, cancel.

The mask is propagated from the anchor frame outward (forward to the range end,
then backward to the start) with optical flow. The scene playhead follows along
so you see each frame in the preview with the current mask outline drawn on top
(via the shared overlay), and you drive it from the keyboard:

    Space   play / pause
    →       step one frame forward
    ←       step back (undo the last tracked frame)
    Enter   finish (keep mattes, add result strip)
    Esc     cancel (keep mattes written so far)

Mattes are written as PNGs while tracking, so partial results always persist.
"""

from __future__ import annotations

import os

import bpy

from . import common
from . import output
from .op_pick import _points_to_pixels, _find_preview_region
from ..engine import loader
from ..engine import maskproc
from ..engine.propagate import _to_gray, _dense_flow, _advect_points
from ..preferences import get_prefs, resolve_output_dir
from ..properties import MODELS
from ..ui import overlay


class SAM2_OT_track(bpy.types.Operator):
    bl_idname = "auto_roto.track"
    bl_label = "Track & Export Matte"
    bl_description = (
        "Interactively propagate the mask across the range. "
        "Space play/pause, arrows step, Enter finish, Esc cancel"
    )
    bl_options = {"REGISTER"}

    _timer = None

    # -- setup --------------------------------------------------------------

    def invoke(self, context, event):
        scene = context.scene
        props = scene.auto_roto

        if len(props.points) == 0:
            self.report({"ERROR"}, "Pick the object first (no points set)")
            return {"CANCELLED"}

        strip = common.active_sequence_strip(context)
        if strip is None:
            self.report({"ERROR"}, "Select a Movie or Image strip in the VSE first")
            return {"CANCELLED"}

        if props.use_scene_range:
            self._start, self._end = scene.frame_start, scene.frame_end
        else:
            self._start, self._end = props.frame_start, props.frame_end
        self._anchor = min(max(props.anchor_frame, self._start), self._end)

        try:
            self._reader = common.FrameReader(strip)
            self._sam = loader.load(MODELS[props.model]["dir"], get_prefs(context).provider_mode)
        except Exception as exc:
            self.report({"ERROR"}, f"Init failed: {exc}")
            return {"CANCELLED"}

        self._out_dir = resolve_output_dir(context)
        self._quality = props.flow_quality
        props.active_provider = loader.active_provider()

        tr = getattr(strip, "transform", None)
        self._ox = getattr(tr, "offset_x", 0.0)
        self._oy = getattr(tr, "offset_y", 0.0)
        self._sx = getattr(tr, "scale_x", 1.0) or 1.0
        self._sy = getattr(tr, "scale_y", 1.0) or 1.0
        rnd = scene.render
        self._rw, self._rh = rnd.resolution_x, rnd.resolution_y

        self._schedule = (
            [("F", f) for f in range(self._anchor + 1, self._end + 1)]
            + [("B", f) for f in range(self._anchor - 1, self._start - 1, -1)]
        )
        self._i = 0
        self._total = len(self._schedule) + 1  # + anchor

        self._labels = _points_to_pixels(props, 1, 1)[1]  # labels only
        self._written = {}
        self._contours_by_frame = {}
        self._history = []
        self._playing = True
        self._finished_all = False
        self._cur_frame = self._anchor
        self._dir = "F"

        try:
            self._process_anchor(context)
        except Exception as exc:
            self.report({"ERROR"}, f"Anchor failed: {exc}")
            self._reader.release()
            return {"CANCELLED"}

        overlay.set_geometry(self._ox, self._oy, self._sx, self._sy, self._w, self._h)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        self._update_display(context)
        return {"RUNNING_MODAL"}

    def _process_anchor(self, context):
        rgb = self._reader.get_frame(self._anchor)
        self._h, self._w = rgb.shape[0], rgb.shape[1]
        coords, _ = _points_to_pixels(context.scene.auto_roto, self._w, self._h)
        emb = self._sam.encode(rgb)
        mask, low, _ = self._sam.decode(emb, coords, self._labels)

        self._prev_gray = _to_gray(rgb)
        self._prev_frame = self._anchor
        self._points = coords
        self._low = low
        self._store_result(context, self._anchor, mask)

    # -- modal loop ---------------------------------------------------------

    def modal(self, context, event):
        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            return self._finish(context, cancelled=False)
        if event.type == "ESC" and event.value == "PRESS":
            return self._finish(context, cancelled=True)
        if event.type == "SPACE" and event.value == "PRESS":
            self._playing = (not self._playing) and not self._finished_all
            self._update_display(context)
            return {"RUNNING_MODAL"}
        if event.type == "RIGHT_ARROW" and event.value == "PRESS":
            self._playing = False
            self._advance(context)
            return {"RUNNING_MODAL"}
        if event.type == "LEFT_ARROW" and event.value == "PRESS":
            self._playing = False
            self._step_back(context)
            return {"RUNNING_MODAL"}

        if event.type == "TIMER":
            if self._playing and not self._finished_all:
                self._advance(context)
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    # -- stepping -----------------------------------------------------------

    def _advance(self, context):
        if self._i >= len(self._schedule):
            self._finished_all = True
            self._playing = False
            self._update_display(context)
            return

        direction, frame = self._schedule[self._i]

        if direction != self._dir:  # entering the backward pass: reset to anchor
            self._dir = direction
            anchor_rgb = self._reader.get_frame(self._anchor)
            self._prev_gray = _to_gray(anchor_rgb)
            self._prev_frame = self._anchor
            coords, _ = _points_to_pixels(context.scene.auto_roto, self._w, self._h)
            self._points = coords
            _, self._low, _ = self._sam.decode(self._sam.encode(anchor_rgb), coords, self._labels)

        self._history.append(
            dict(
                i=self._i,
                dir=self._dir,
                prev_frame=self._prev_frame,
                points=self._points.copy(),
                low=self._low.copy(),
            )
        )

        try:
            rgb = self._reader.get_frame(frame)
            gray = _to_gray(rgb)
            flow = _dense_flow(self._prev_gray, gray, self._quality)
            self._points = _advect_points(self._points, flow)
            emb = self._sam.encode(rgb)
            mask, self._low, _ = self._sam.decode(emb, self._points, self._labels, mask_input=self._low)
            self._prev_gray = gray
            self._prev_frame = frame
        except Exception as exc:
            self._history.pop()
            self.report({"WARNING"}, f"Frame {frame} failed: {exc}")
            self._playing = False
            self._update_display(context)
            return

        self._i += 1
        self._store_result(context, frame, mask)
        if self._i >= len(self._schedule):
            self._finished_all = True
            self._playing = False
        self._update_display(context)

    def _step_back(self, context):
        if not self._history:
            return
        snap = self._history.pop()
        undone_frame = self._schedule[snap["i"]][1]

        path = self._written.pop(undone_frame, None)
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
        self._contours_by_frame.pop(undone_frame, None)

        self._i = snap["i"]
        self._dir = snap["dir"]
        self._prev_frame = snap["prev_frame"]
        self._points = snap["points"]
        self._low = snap["low"]
        self._prev_gray = _to_gray(self._reader.get_frame(self._prev_frame))
        self._finished_all = False

        prev_kept = self._schedule[snap["i"] - 1][1] if snap["i"] > 0 else self._anchor
        self._cur_frame = prev_kept
        context.scene.frame_current = prev_kept
        self._update_display(context)

    def _store_result(self, context, frame, mask):
        props = context.scene.auto_roto
        refined = maskproc.refine_mask(mask, props.single_shape)
        alpha = maskproc.feather_alpha(refined, props.feather)
        placement = (self._rw, self._rh, self._ox, self._oy, self._sx, self._sy)
        self._written[frame] = output.write_matte(self._out_dir, frame, alpha, placement)
        self._contours_by_frame[frame] = maskproc.mask_to_contours(refined)
        self._cur_frame = frame
        context.scene.frame_current = frame

    # -- display ------------------------------------------------------------

    def _update_display(self, context):
        overlay.set_contours(self._contours_by_frame.get(self._cur_frame, []))
        done = len(self._written)
        state = "DONE" if self._finished_all else ("PLAYING" if self._playing else "PAUSED")
        overlay.set_hud(
            f"Tracking {done}/{self._total}  [{state}]   "
            f"Space play/pause  <-/-> step  Enter finish  Esc cancel"
        )
        context.scene.auto_roto.status = f"Tracking {done}/{self._total} ({state.lower()})"
        overlay.tag_redraw_previews(context)

    # -- teardown -----------------------------------------------------------

    def _finish(self, context, cancelled):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        self._reader.release()
        overlay.clear()
        overlay.tag_redraw_previews(context)

        props = context.scene.auto_roto
        n = len(self._written)
        if props.add_strip and self._written:
            try:
                output.rebuild_result_strip(context, self._out_dir, self._written.keys())
            except Exception as exc:
                self.report({"WARNING"}, f"Could not add result strip: {exc}")
        if cancelled:
            props.status = f"Tracking cancelled — {n} matte(s) kept in {self._out_dir}"
        else:
            props.status = f"Done: {n} matte(s) → {self._out_dir}"
        self.report({"INFO"}, props.status)
        return {"CANCELLED"} if cancelled else {"FINISHED"}


def register():
    bpy.utils.register_class(SAM2_OT_track)


def unregister():
    bpy.utils.unregister_class(SAM2_OT_track)
