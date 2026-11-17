"""Render Reference output with each light sampling mode using the embedded backend."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if '--installed' not in sys.argv:
    sys.path[:0] = [str(ROOT), str(ROOT / 'photara/python')]
import addon_utils
import bpy
import numpy as np
import _photara_native

addon_utils.enable('photara_rs_engine')
from photara_rs_engine import native_backend, native_engine
assert native_backend.available()
scene = bpy.context.scene
scene.render.engine = 'PHOTARA_RS'
scene.render.resolution_x = 96
scene.render.resolution_y = 72
scene.render.resolution_percentage = 100
scene.render.use_persistent_data = True
scene.render.image_settings.file_format = 'PNG'
settings = scene.photara_rs
settings.render_mode = 'REFERENCE'
settings.samples = 16
settings.denoise = False
settings.adaptive_sampling = False
output = ROOT / 'test_outputs/reference_light_sampling'
output.mkdir(parents=True, exist_ok=True)
reports = []
original = native_engine.render


def capture(engine, depsgraph):
    original(engine, depsgraph)
    reports.append(dict(_photara_native.session_stats(engine.session)))


native_engine.render = capture
for enabled, mode in ((False, 'HYBRID'), (True, 'PURE'), (True, 'HYBRID'), (False, 'PURE')):
    settings.realtime_neeat = enabled
    settings.realtime_neeat_mode = mode
    scene.render.filepath = str(output / f'{len(reports)}.png')
    bpy.ops.render.render(write_still=True)
    stats = reports[-1]
    assert (stats['neeat_emitters'] > 0) == enabled, stats
    assert abs(stats['neeat_tree_fraction'] - (0 if mode == 'PURE' else .25)) < 1e-6, stats
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    pixels = np.array(image.pixels[:])
    bpy.data.images.remove(image)
    assert np.isfinite(pixels).all() and pixels.reshape(-1, 4)[:, :3].mean() > .001
print('REFERENCE_LIGHT_SAMPLING_PASS', json.dumps(reports), flush=True)
