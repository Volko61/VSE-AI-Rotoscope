"""Single persistent preview overlay: point markers, mask outline, HUD text.

Centralising the draw callback here (instead of each operator adding its own)
avoids leaked/stacked handlers and lets point deletion update instantly — the
markers are read live from the scene, so a redraw is all that's needed.
"""

from __future__ import annotations

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader

_handle = None

# Geometry of the tracked strip in the preview (set when picking/tracking).
_state = {
    "ox": 0.0, "oy": 0.0, "sx": 1.0, "sy": 1.0, "w": 1.0, "h": 1.0,
    "contours": [],   # list of (N,2) arrays in source-pixel coords
    "hud": "",
}


def set_geometry(ox, oy, sx, sy, w, h):
    _state.update(ox=ox, oy=oy, sx=sx, sy=sy, w=float(w), h=float(h))


def set_contours(contours):
    _state["contours"] = contours or []


def set_hud(text):
    _state["hud"] = text or ""


def clear():
    _state["contours"] = []
    _state["hud"] = ""


def tag_redraw_previews(context=None):
    context = context or bpy.context
    win = getattr(context, "window", None)
    screens = [win.screen] if win else [w.screen for w in bpy.data.window_managers[0].windows]
    for screen in screens:
        for area in screen.areas:
            if area.type == "SEQUENCE_EDITOR":
                area.tag_redraw()


def _src_to_region(region, px, py):
    nx = px / _state["w"]
    ny = py / _state["h"]
    vx = _state["ox"] + (nx - 0.5) * _state["w"] * _state["sx"]
    vy = _state["oy"] + (0.5 - ny) * _state["h"] * _state["sy"]
    return region.view2d.view_to_region(vx, vy, clip=False)


def _draw():
    region = bpy.context.region
    if region is None or region.type != "PREVIEW" or region.width <= 1:
        return
    scene = getattr(bpy.context, "scene", None)
    if scene is None or not hasattr(scene, "auto_roto"):
        return
    props = scene.auto_roto

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")

    # Mask outline (cyan).
    gpu.state.line_width_set(2.0)
    for cnt in _state["contours"]:
        pts = [_src_to_region(region, x, y) for (x, y) in cnt]
        if len(pts) >= 2:
            batch = batch_for_shader(shader, "LINE_LOOP", {"pos": pts})
            shader.bind()
            shader.uniform_float("color", (0.1, 0.9, 1.0, 0.95))
            batch.draw(shader)

    # Point markers: green = positive, red = negative.
    pos_pts, neg_pts = [], []
    for p in props.points:
        r = _src_to_region(region, p.x * _state["w"], (1.0 - p.y) * _state["h"])
        (pos_pts if p.positive else neg_pts).append(r)

    gpu.state.point_size_set(10.0)
    for pts, color in ((pos_pts, (0.1, 1.0, 0.2, 1.0)), (neg_pts, (1.0, 0.15, 0.15, 1.0))):
        if pts:
            batch = batch_for_shader(shader, "POINTS", {"pos": pts})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

    gpu.state.blend_set("NONE")

    if _state["hud"]:
        blf.position(0, 20, region.height - 40, 0)
        blf.size(0, 16)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        blf.draw(0, _state["hud"])


def register():
    global _handle
    if _handle is None:
        _handle = bpy.types.SpaceSequenceEditor.draw_handler_add(
            _draw, (), "PREVIEW", "POST_PIXEL"
        )


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceSequenceEditor.draw_handler_remove(_handle, "PREVIEW")
        _handle = None
    clear()
