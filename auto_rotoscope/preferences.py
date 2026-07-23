"""Add-on preferences: execution provider, output folder, engine status."""

import os

import bpy

from .engine import ort_session


class SAM2RotoPreferences(bpy.types.AddonPreferences):
    # Must match the extension id (package name).
    bl_idname = __package__

    provider_mode: bpy.props.EnumProperty(
        name="Compute Device",
        description="Which device onnxruntime should use for inference",
        items=[
            ("AUTO", "Auto", "Use the GPU if available, otherwise CPU"),
            ("CPU", "Force CPU", "Always run on CPU"),
            ("GPU", "Force GPU", "Use the GPU (DirectML on Windows); falls back to CPU on failure"),
        ],
        default="AUTO",
    )

    output_dir: bpy.props.StringProperty(
        name="Output Folder",
        description="Where matte sequences are written. Leave empty to use a folder next to the .blend",
        subtype="DIR_PATH",
        default="",
    )

    def draw(self, context):
        layout = self.layout

        col = layout.column()
        col.prop(self, "provider_mode")
        col.prop(self, "output_dir")

        box = layout.box()
        box.label(text="Engine status", icon="INFO")
        if ort_session.is_available():
            provs = ", ".join(ort_session.available_providers()) or "none"
            box.label(text=f"onnxruntime OK — providers: {provs}")
            box.label(
                text=("GPU available" if ort_session.has_gpu() else "CPU only"),
                icon=("CHECKMARK" if ort_session.has_gpu() else "DOT"),
            )
        else:
            box.label(text="onnxruntime failed to load:", icon="ERROR")
            box.label(text=ort_session.import_error()[:120])


def get_prefs(context) -> SAM2RotoPreferences:
    return context.preferences.addons[__package__].preferences


def resolve_output_dir(context) -> str:
    """Return an absolute output directory, creating a sensible default."""
    prefs = get_prefs(context)
    if prefs.output_dir:
        path = bpy.path.abspath(prefs.output_dir)
    else:
        base = bpy.path.abspath("//") or os.path.join(os.path.expanduser("~"), "auto_roto")
        path = os.path.join(base, "auto_roto_mattes")
    os.makedirs(path, exist_ok=True)
    return path


def register():
    bpy.utils.register_class(SAM2RotoPreferences)


def unregister():
    bpy.utils.unregister_class(SAM2RotoPreferences)
