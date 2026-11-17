"""Blender output/view-layer panel lifecycle and native multi-pass rendering."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if '--installed' not in sys.argv:
    sys.path[:0] = [str(ROOT), str(ROOT / 'photara/python')]
import bpy
import addon_utils
import numpy as np
from bl_ui import properties_output, properties_view_layer
from cycles import ui as cycles_ui

shared = [cls for module in (properties_output, properties_view_layer)
          for cls in module.classes if hasattr(cls, 'COMPAT_ENGINES')]
before = {cls: set(cls.COMPAT_ENGINES) for cls in shared}
cycles_before = set(cycles_ui.CyclesButtonsPanel.COMPAT_ENGINES)
addon_utils.enable('photara_rs_engine')
from photara_rs_engine import ui, native_engine
scene = bpy.context.scene
scene.render.engine = 'PHOTARA_RS'
for name in ('RENDER_PT_format', 'RENDER_PT_frame_range', 'RENDER_PT_output',
             'RENDER_PT_post_processing'):
    assert getattr(bpy.types, name).poll(bpy.context), name
for name in ('VIEWLAYER_PT_layer', 'VIEWLAYER_PT_override'):
    assert getattr(bpy.types, name).poll(bpy.context), name
assert cycles_ui.CyclesButtonsPanel.COMPAT_ENGINES == cycles_before
assert len(ui._view_layer_panels) >= 7


class Layout:
    def __init__(self):
        self.properties = []

    def prop(self, data, name, **kwargs):
        assert hasattr(data, name), (data, name)
        self.properties.append((data, name))

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


layout = Layout()
for cls in ui._view_layer_panels:
    assert cls.poll(bpy.context), cls.__name__
    cls.draw(SimpleNamespace(layout=layout), bpy.context)
assert any(data == scene.view_layers[0] and name == 'use_pass_z'
           for data, name in layout.properties)
assert any(data == scene.view_layers[0].cycles and name == 'use_denoising'
           for data, name in layout.properties)

scene.render.resolution_x = 128
scene.render.resolution_y = 96
scene.render.resolution_percentage = 50
scene.render.fps = 24
if hasattr(scene.render.image_settings, 'media_type'):
    scene.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
scene.render.image_settings.color_depth = '32'
scene.render.use_single_layer = False
scene.photara_rs.samples = 8
scene.photara_rs.denoise = False
scene.photara_rs.adaptive_sampling = False
layer = scene.view_layers[0]
layer.use_pass_z = True
layer.use_pass_normal = True
layer.use_pass_diffuse_direct = True
layer.use_pass_object_index = True
layer.use_pass_cryptomatte_object = True
layer.aovs.add().name = 'SharedAOV'
layer.lightgroups.add(name='SharedLights')
scene.objects['Light'].lightgroup = 'SharedLights'
other = scene.view_layers.new('SharedOverride')
other.use_pass_z = True
material = bpy.data.materials.new('ViewLayerOverride')
material.diffuse_color = (.02, .1, .9, 1)
material.use_nodes = True
material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (.02, .1, .9, 1)
other.material_override = material
out = ROOT / 'test_outputs/output_view_layer'
out.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(out / 'layers.exr')
reports = []
original = native_engine.render


def capture(engine, depsgraph):
    original(engine, depsgraph)
    result = engine.get_result()
    reports.append(dict(width=result.resolution_x, height=result.resolution_y,
                        layers={layer.name: [p.name for p in layer.passes] for layer in result.layers},
                        mean_rgb={layer.name: np.asarray(layer.passes['Combined'].rect).reshape(-1, 4)[:, :3].mean(axis=0).tolist()
                                  for layer in result.layers}))


native_engine.render = capture
bpy.ops.render.render(write_still=True)
assert (out / 'layers.exr').is_file()
report = reports[-1]
assert (report['width'], report['height']) == (64, 48), report
passes = report['layers'][layer.name]
for name in ('Combined', 'Depth', 'Normal', 'Diffuse Direct', 'Object Index', 'CryptoObject00',
             'SharedAOV', 'Combined_SharedLights'):
    assert name in passes, (name, report)
assert 'SharedOverride' in report['layers'], report
assert 'Depth' in report['layers']['SharedOverride'], report
override_rgb = report['mean_rgb']['SharedOverride']
assert np.isfinite(override_rgb).all() and override_rgb[2] > override_rgb[0], report
# Verify the values survive switching engines; the UI edits Blender's own RNA.
scene.render.engine = 'CYCLES'
assert layer.use_pass_z and layer.use_pass_normal
assert scene.render.resolution_percentage == 50
assert other.material_override == material
scene.render.engine = 'BLENDER_EEVEE'
addon_utils.disable('photara_rs_engine')
for cls, engines in before.items():
    assert cls.COMPAT_ENGINES == engines, cls.__name__
assert cycles_ui.CyclesButtonsPanel.COMPAT_ENGINES == cycles_before
addon_utils.enable('photara_rs_engine')
assert len(ui._view_layer_panels) >= 7
(out / 'verification.json').write_text(json.dumps(report, indent=2))
print('OUTPUT_VIEW_LAYER_PASS', json.dumps(report), flush=True)
