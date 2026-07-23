"""Scene-level state for a rotoscoping session."""

import bpy


# Model registry. Only 'tiny' is bundled (keeps the ZIP under the 200 MB
# platform limit and fully offline). The structure is intentionally extensible.
MODELS = {
    "tiny": {
        "label": "Hiera Tiny (bundled)",
        "dir": "sam2.1_hiera_tiny",
    },
}


def model_enum_items(self, context):
    return [(key, m["label"], "") for key, m in MODELS.items()]


class SAM2RotoPoint(bpy.types.PropertyGroup):
    # Normalized image coordinates (0..1), so they survive resolution changes.
    x: bpy.props.FloatProperty(name="X", default=0.0)
    y: bpy.props.FloatProperty(name="Y", default=0.0)
    positive: bpy.props.BoolProperty(
        name="Positive",
        description="Positive point adds to the selection, negative removes",
        default=True,
    )


class SAM2RotoProps(bpy.types.PropertyGroup):
    model: bpy.props.EnumProperty(
        name="Model",
        description="SAM 2.1 model variant",
        items=model_enum_items,
    )

    points: bpy.props.CollectionProperty(type=SAM2RotoPoint)
    active_point: bpy.props.IntProperty(name="Active Point", default=0)

    anchor_frame: bpy.props.IntProperty(
        name="Anchor Frame",
        description="Frame where the object was picked; tracking propagates outward from here",
        default=1,
    )

    add_strip: bpy.props.BoolProperty(
        name="Add Result Strip",
        description="Add the generated matte sequence as an image strip in the VSE",
        default=True,
    )

    use_scene_range: bpy.props.BoolProperty(
        name="Use Scene Frame Range",
        description="Track over the scene's start/end frames",
        default=True,
    )
    frame_start: bpy.props.IntProperty(name="Start", default=1, min=0)
    frame_end: bpy.props.IntProperty(name="End", default=250, min=0)

    single_shape: bpy.props.BoolProperty(
        name="Single Shape",
        description="Keep only the largest region and fill its holes — one closed shape, no stray islands",
        default=False,
    )

    feather: bpy.props.FloatProperty(
        name="Feather",
        description="Soften the matte edges by this radius (pixels)",
        default=0.0,
        min=0.0,
        max=100.0,
        subtype="PIXEL",
    )

    flow_quality: bpy.props.EnumProperty(
        name="Tracking Quality",
        description="Optical-flow accuracy vs. speed for propagation",
        items=[
            ("FAST", "Fast", "Coarser flow, quicker"),
            ("BALANCED", "Balanced", "Good default"),
            ("ACCURATE", "Accurate", "Finer flow, slower"),
        ],
        default="BALANCED",
    )

    # Runtime status (updated by operators; read-only in the UI).
    status: bpy.props.StringProperty(name="Status", default="")
    active_provider: bpy.props.StringProperty(name="Provider", default="")


_classes = (SAM2RotoPoint, SAM2RotoProps)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
