"""Continuous object/deformation history, disocclusion and edit invalidation.

Run in the rebuilt Blender; optionally open a scene and pass -- --scene-test.
"""
import json
from pathlib import Path
import sys
import time
import bpy
import addon_utils
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if '--installed' not in sys.argv:
    sys.path[:0] = [str(ROOT), str(ROOT / 'photara/python')]
addon_utils.enable('photara_rs_engine')
from photara_rs_engine import native_engine
import _photara_native

SCENE_TEST = '--scene-test' in sys.argv
scene = bpy.context.scene
if not SCENE_TEST:
    cube = scene.objects['Cube']
    cube.location.x = -.6
    cube.keyframe_insert('location', frame=1)
    cube.location.x = .6
    cube.keyframe_insert('location', frame=3)
    cube.shape_key_add(name='Basis')
    deform = cube.shape_key_add(name='Deform')
    for vertex in deform.data:
        vertex.co.z += .25 * max(vertex.co.x, 0)
    deform.value = 0
    deform.keyframe_insert('value', frame=1)
    deform.value = 1
    deform.keyframe_insert('value', frame=3)
    material = bpy.data.materials.new('Dynamic red')
    material.use_nodes = True
    material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (.8, .03, .02, 1)
    material.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = .8
    cube.data.materials.clear()
    cube.data.materials.append(material)
scene.render.engine = 'PHOTARA_RS'
p = scene.photara_rs
p.render_mode = 'REALTIME'
p.samples = 8
p.history_samples = 8
p.realtime_warmup_frames = 8
p.realtime_neeat = True
p.realtime_neeat_mode = 'HYBRID'
p.denoise = SCENE_TEST
p.denoiser = 'DLSS'
p.denoise_upscale_quality = 'NONE'
scene.render.resolution_x = 640 if SCENE_TEST else 160
scene.render.resolution_y = 360 if SCENE_TEST else 120
scene.render.resolution_percentage = 100
scene.render.use_motion_blur = False
scene.render.use_border = False
scene.render.use_persistent_data = True
scene.render.use_compositing = False
scene.render.use_sequencer = False
scene.render.image_settings.media_type = 'IMAGE'
scene.render.image_settings.file_format = 'PNG'
scene.frame_start = 141 if SCENE_TEST else 1
scene.frame_end = scene.frame_start + 3
scene.frame_step = 1
output = ROOT / 'test_outputs' / ('apple_dynamic_history' if SCENE_TEST else 'dynamic_history')
output.mkdir(exist_ok=True)
scene.render.filepath = str(output / 'continuous_')
reports = []
original = native_engine.render


def capture(engine, depsgraph):
    start = time.monotonic()
    original(engine, depsgraph)
    stats = dict(_photara_native.session_stats(engine.session))
    stats.update(history=dict(_photara_native.realtime_stats(engine.session)),
                 frame=scene.frame_current, revision=p.history_revision,
                 elapsed_seconds=time.monotonic()-start)
    result = engine.get_result()
    color = np.asarray(result.layers[0].passes['Combined'].rect).reshape(-1, 4)
    assert np.isfinite(color).all() and color[:, :3].std() > .001, stats
    reports.append(stats)
    (output / 'verification.json').write_text(json.dumps(reports, indent=2))
    print('DYNAMIC_HISTORY_STATS', json.dumps(stats), flush=True)


native_engine.render = capture
bpy.ops.render.render(animation=True)
assert len(reports) == 4, reports
for previous, current in zip(reports, reports[1:]):
    assert current['revision'] == previous['revision'], reports
    assert current['history']['gpu_updates'] - previous['history']['gpu_updates'] == 1, reports
    assert current['history']['cross_frame_pixels'] > 0, current
assert any(r['history']['moving_pixels'] > 0 for r in reports[1:]), reports
if SCENE_TEST:
    assert all(r['denoiser_type'] == 8 and r['use_denoise'] for r in reports), reports
    assert reports[-1]['dlss_evaluations'] - reports[0]['dlss_evaluations'] == 3, reports
    # Compare the last continuous frame with an independently warmed render of
    # the same geometry. This catches gross stale-frame contamination.
    continuous = bpy.data.images.load(str(output / f'continuous_{scene.frame_end:04d}.png'), check_existing=False)
    continuous_pixels = np.asarray(continuous.pixels[:]).reshape(-1, 4)[:, :3]
    bpy.data.images.remove(continuous)
    scene.frame_set(scene.frame_end)
    p.history_revision += 1
    scene.render.filepath = str(output / 'reset_reference.png')
    bpy.ops.render.render(write_still=True)
    independent = bpy.data.images.load(scene.render.filepath, check_existing=False)
    independent_pixels = np.asarray(independent.pixels[:]).reshape(-1, 4)[:, :3]
    bpy.data.images.remove(independent)
    rmse = float(np.sqrt(np.mean((continuous_pixels-independent_pixels)**2)))
    assert rmse < .1, rmse
    (output / 'comparison.json').write_text(json.dumps(dict(reset_reference_rmse=rmse)))
else:
    # In-place material edits must still invalidate geometry history.
    revision = p.history_revision
    material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (.02, .1, .8, 1)
    scene.render.filepath = str(output / 'material_edit.png')
    bpy.ops.render.render(write_still=True)
    assert p.history_revision > revision, reports[-1]
print('DYNAMIC_HISTORY_PASS', json.dumps(dict(output=str(output),scene_test=SCENE_TEST)), flush=True)
