"""Clear the current picking session (points + overlay)."""

import bpy

from ..ui import overlay


class SAM2_OT_clear(bpy.types.Operator):
    bl_idname = "auto_roto.clear"
    bl_label = "Clear Points"
    bl_description = "Remove all picked points and reset the session"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.auto_roto
        props.points.clear()
        props.active_point = 0
        props.status = "Cleared"
        overlay.clear()
        overlay.tag_redraw_previews(context)
        return {"FINISHED"}


class SAM2_OT_remove_point(bpy.types.Operator):
    bl_idname = "auto_roto.remove_point"
    bl_label = "Remove Point"
    bl_description = "Remove the selected point"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(default=-1)

    def execute(self, context):
        props = context.scene.auto_roto
        idx = self.index if self.index >= 0 else props.active_point
        if 0 <= idx < len(props.points):
            props.points.remove(idx)
            props.active_point = min(props.active_point, len(props.points) - 1)
        overlay.set_contours([])  # stale outline no longer matches the points
        overlay.tag_redraw_previews(context)
        return {"FINISHED"}


_classes = (SAM2_OT_clear, SAM2_OT_remove_point)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
