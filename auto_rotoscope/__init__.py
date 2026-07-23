"""SAM2 Roto — automatic rotoscoping for the Blender Video Sequence Editor.

Runs SAM 2.1 (image encoder/decoder) through onnxruntime — no PyTorch — and
propagates the selected mask across a frame range with OpenCV optical flow.
Everything is bundled and runs fully offline (CPU by default, GPU when available).
"""

import bpy

from . import properties
from . import preferences
from .ops import op_pick, op_track, op_clear
from .ui import panel_vse, overlay

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_modules = (
    properties,
    preferences,
    op_pick,
    op_track,
    op_clear,
    panel_vse,
    overlay,
)

_addon_keymaps = []


def register():
    for mod in _modules:
        mod.register()

    # Scene-level property group holding the roto session state.
    bpy.types.Scene.auto_roto = bpy.props.PointerProperty(type=properties.SAM2RotoProps)

    # Keymap: press the pick shortcut inside the Image Editor.
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name="SequencerPreview", space_type="SEQUENCE_EDITOR")
        kmi = km.keymap_items.new(op_pick.SAM2_OT_pick.bl_idname, "L", "PRESS")
        _addon_keymaps.append((km, kmi))


def unregister():
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()

    del bpy.types.Scene.auto_roto

    for mod in reversed(_modules):
        mod.unregister()
