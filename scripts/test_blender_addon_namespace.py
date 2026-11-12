"""Run in background Blender: settings migration and add-on coexistence."""
from pathlib import Path
import sys
import tempfile

import addon_utils
import bpy

# Exercise the checkout rather than an older installed renderer add-on.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def enable(name):
    module = addon_utils.enable(name)
    assert module is not None, f'Could not enable {name}'
    return module


def configure(renderer, scene):
    assert scene.photara_rs.bl_rna.identifier == 'PhotaraSceneSettings'
    scene.photara_rs.samples = 7
    scene.photara_rs.viewport_samples = 3
    scene.photara_rs.denoise = False
    if renderer.native_backend.available():
        renderer.native_backend.configure(scene)
        assert scene.photara_native.samples == 7
        assert scene.photara_native.preview_samples == 3


scene = bpy.context.scene
for name in ('photara_rs_engine', 'photara_render'):
    if addon_utils.check(name)[1]:
        addon_utils.disable(name)

# Old blend files store the original PropertyGroup as ID properties, including
# numeric enum values. Registration and subsequent loads must preserve these.
import photara_rs_engine
bpy.utils.register_class(photara_rs_engine.properties.PhotaraSceneSettings)
bpy.types.Scene.photara = bpy.props.PointerProperty(
    type=photara_rs_engine.properties.PhotaraSceneSettings)
scene.photara.samples = 19
scene.photara.viewport_samples = 5
scene.photara.denoise = False
scene.photara.denoiser = 'DLSS'
scene.photara['dataset_name'] = 'unrelated-export-data'
output = Path(tempfile.gettempdir()) / 'photara_namespace_legacy.blend'
bpy.ops.wm.save_as_mainfile(filepath=str(output))
del bpy.types.Scene.photara
bpy.utils.unregister_class(photara_rs_engine.properties.PhotaraSceneSettings)
renderer = enable('photara_rs_engine')
# Background scripts run before the UI timer loop; invoke the queued migration.
renderer.properties.migrate_legacy_settings()
assert scene.photara_rs.samples == 19
assert scene.photara_rs.viewport_samples == 5
assert scene.photara_rs.denoise is False
assert scene.photara_rs.denoiser == 'DLSS'
assert 'dataset_name' not in scene.photara_rs
scene.photara_rs.samples = 23
renderer.properties.migrate_legacy_settings()
assert scene.photara_rs.samples == 23, 'Migration overwrote new settings'
bpy.ops.wm.open_mainfile(filepath=str(output))
scene = bpy.context.scene
assert scene.photara_rs.samples == 19, 'load_post did not migrate settings'

# Use the actual installed dataset add-on when present. A conflicting schema
# also makes the test useful on machines without that optional add-on.
try:
    import photara_render
except ImportError:
    class PhotaraProps(bpy.types.PropertyGroup):
        dataset_name: bpy.props.StringProperty()

    class DatasetAddon:
        @staticmethod
        def register():
            bpy.utils.register_class(PhotaraProps)
            bpy.types.Scene.photara = bpy.props.PointerProperty(type=PhotaraProps)

        @staticmethod
        def unregister():
            del bpy.types.Scene.photara
            bpy.utils.unregister_class(PhotaraProps)

    photara_render = DatasetAddon()

photara_render.register()
# File loading must also migrate when the legacy name is already owned by the
# dataset exporter, without replacing that exporter's RNA property.
bpy.ops.wm.open_mainfile(filepath=str(output))
scene = bpy.context.scene
assert scene.photara.bl_rna.identifier == 'PhotaraProps'
assert scene.photara_rs.samples == 19
scene.photara.dataset_name = 'coexistence'
configure(renderer, scene)
addon_utils.disable('photara_rs_engine')
assert scene.photara.dataset_name == 'coexistence', 'Renderer removed exporter settings'
renderer = enable('photara_rs_engine')
configure(renderer, scene)
assert scene.photara.dataset_name == 'coexistence'
photara_render.unregister()
configure(renderer, scene)
addon_utils.disable('photara_rs_engine')
assert renderer.properties.migrate_legacy_settings not in bpy.app.handlers.load_post
assert not bpy.app.timers.is_registered(renderer.properties.migrate_legacy_settings)
output.unlink()
print('PHOTARA_NAMESPACE_OK: migration, both load orders, independent teardown')
