"""Sidebar panel in the Video Sequence Editor."""

import bpy

from ..engine import loader
from ..preferences import get_prefs


class SAM2_UL_points(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index):
        row = layout.row(align=True)
        row.label(
            text=f"{'＋' if item.positive else '－'}  ({item.x:.2f}, {item.y:.2f})",
            icon="ADD" if item.positive else "REMOVE",
        )
        op = row.operator("auto_roto.remove_point", text="", icon="X", emboss=False)
        op.index = index


class SAM2_PT_panel(bpy.types.Panel):
    bl_label = "SAM2 Roto"
    bl_idname = "SAM2_PT_panel"
    bl_space_type = "SEQUENCE_EDITOR"
    bl_region_type = "UI"
    bl_category = "SAM2 Roto"

    def draw(self, context):
        layout = self.layout
        props = context.scene.auto_roto
        prefs = get_prefs(context)

        # Device / engine status.
        row = layout.row()
        gpu = loader.active_provider() or ("GPU" if _gpu(context) else "CPU")
        row.label(text=f"Device: {prefs.provider_mode}", icon="MEMORY")
        row.label(text=props.active_provider or gpu)

        layout.prop(props, "model")

        # Picking.
        box = layout.box()
        box.label(text="1 · Pick object", icon="RESTRICT_SELECT_OFF")
        box.operator("auto_roto.pick", icon="EYEDROPPER")

        r = box.row()
        r.template_list("SAM2_UL_points", "", props, "points", props, "active_point", rows=3)
        col = box.column(align=True)
        col.operator("auto_roto.clear", text="", icon="TRASH")
        box.label(text=f"Points: {len(props.points)}")

        # Mask shaping (applies to preview + exported mattes).
        box = layout.box()
        box.label(text="2 · Mask", icon="MOD_MASK")
        box.prop(props, "single_shape")
        box.prop(props, "feather")

        # Range + tracking.
        box = layout.box()
        box.label(text="3 · Track range", icon="TRACKING")
        box.prop(props, "use_scene_range")
        if not props.use_scene_range:
            row = box.row(align=True)
            row.prop(props, "frame_start")
            row.prop(props, "frame_end")
        box.prop(props, "flow_quality")
        box.prop(props, "add_strip")
        box.operator("auto_roto.track", icon="RENDER_ANIMATION")
        col = box.column(align=True)
        col.scale_y = 0.8
        col.label(text="Space play/pause · ←/→ step")
        col.label(text="Enter finish · Esc cancel")

        # Status line.
        if props.status:
            layout.separator()
            layout.label(text=props.status, icon="INFO")


def _gpu(context):
    try:
        return loader.ort_session.has_gpu()  # type: ignore[attr-defined]
    except Exception:
        return False


_classes = (SAM2_UL_points, SAM2_PT_panel)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
