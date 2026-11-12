# SPDX-License-Identifier: Apache-2.0
#
# Blender add-on for the Photara renderer.
#
# The add-on registers a `PHOTARA_RS` render engine and drives it through the
# `photara_rs` Python package (the same wheel used for standalone Python
# workflows). No Blender build changes are required: install the wheel into
# Blender's Python, then install this add-on like any other extension.

bl_info = {
    "name": "Photara Renderer",
    "author": "Photara contributors",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "Render Properties > Photara",
    "description": "Render with the Photara OptiX path tracer",
    "doc_url": "",
    "tracker_url": "",
    "category": "Render",
}

import bpy

from . import engine, properties, sync, ui, gsplat_import, native_backend


def ensure_photara_module():
    """Make `photara_rs` importable from the add-on preferences path.

    Returns None on success or an error message.
    """
    import sys
    import os

    prefs = bpy.context.preferences.addons.get(__package__)
    module_path = ""
    if prefs is not None:
        module_path = getattr(prefs.preferences, "module_path", "")
    if module_path and os.path.isdir(module_path) and module_path not in sys.path:
        sys.path.insert(0, module_path)

    try:
        import photara_rs  # noqa: F401
        engine.photara_rs = photara_rs
        return None
    except Exception as exc:
        return str(exc)


class PhotaraAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    module_path: bpy.props.StringProperty(
        name="photara_rs Path",
        description=(
            "Optional directory containing the photara_rs package or the "
            "installed wheel (added to sys.path before importing). Leave empty "
            "when photara_rs is installed in Blender's Python"
        ),
        subtype='NONE',
    )  # type: ignore[attr-defined]

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "module_path")

        error = ensure_photara_module()
        if error is None:
            import photara_rs

            layout.label(
                text=f"photara_rs {photara_rs.__version__} loaded",
                icon='CHECKMARK',
            )
        else:
            box = layout.box()
            box.label(text="Could not import photara_rs:", icon='ERROR')
            box.label(text=error)
            box.label(
                text=(
                    "Install the wheel into Blender's Python, e.g. "
                    'Run: "Blender Python/python.exe" -m pip install <matching wheel>'
                )
            )


_classes = (
    properties.PhotaraSceneSettings,
    gsplat_import.IMPORT_SCENE_OT_photara_gaussians,
    *ui.classes,
    engine.PhotaraRenderEngine,
    PhotaraAddonPreferences,
)


def register():
    native_backend.register()
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.photara = bpy.props.PointerProperty(
        type=properties.PhotaraSceneSettings
    )
    bpy.types.RENDER_PT_context.append(ui.draw_device)
    ui.register_color_management()
    bpy.types.TOPBAR_MT_file_import.append(gsplat_import.menu_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(gsplat_import.menu_import)
    ui.unregister_color_management()
    bpy.types.RENDER_PT_context.remove(ui.draw_device)
    # Blender 5.x may already have released owner-tagged properties at
    # shutdown; guard so disabling/unloading never raises.
    if hasattr(bpy.types.Scene, "photara"):
        del bpy.types.Scene.photara
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    native_backend.unregister()
