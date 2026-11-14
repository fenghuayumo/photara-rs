"""Run in the rebuilt Blender: RealTime still, image sequence, and encoded video."""
import json
from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[1]
if '--installed' not in sys.argv:
    sys.path[:0] = [str(ROOT), str(ROOT / 'photara/python')]
sys.path.insert(0, str(ROOT / 'scripts'))
import photara_rs_engine
import addon_utils
addon_utils.enable('photara_rs_engine', default_set=True)
from photara_rs_engine import native_engine
import _photara_native
print('REALTIME_ADDON_PATH', photara_rs_engine.__file__, flush=True)

from test_blender import prepare_scene
scene, cube, material = prepare_scene()
settings = scene.photara_rs
scene.render.resolution_x = 96
scene.render.resolution_y = 72
settings.render_mode = 'REALTIME'
settings.samples = 4
settings.history_samples = 2
settings.realtime_warmup_frames = 2
settings.denoise = False
out = ROOT / 'test_outputs/blender_realtime'
out.mkdir(parents=True, exist_ok=True)
captures = []
original = native_engine.render


def capture(engine, depsgraph):
    original(engine, depsgraph)
    stats = dict(_photara_native.realtime_stats(engine.session))
    captures.append(stats)
    assert stats['reconstruction_pixel_readback_bytes'] == 0, stats
    assert stats['samples'] == settings.samples, stats
    print('REALTIME_STATS', json.dumps(stats), flush=True)


native_engine.render = capture
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(out / 'still.png')
bpy.ops.render.render(write_still=True)
assert (out / 'still.png').exists()
assert captures[-1]['gpu_updates'] >= 3, captures
assert captures[-1]['max_history_frames'] == 2, captures
scene.camera.keyframe_insert(data_path='location', frame=1)
scene.camera.location.x += 0.05
scene.camera.keyframe_insert(data_path='location', frame=3)
scene.frame_start, scene.frame_end = 1, 3
scene.render.filepath = str(out / 'sequence_')
captures.clear()
bpy.ops.render.render(animation=True)
assert len(captures) == 3, captures
assert all((out / f'sequence_{i:04d}.png').exists() for i in range(1, 4))
assert captures[-1]['gpu_updates'] - captures[0]['gpu_updates'] == 2, captures
assert captures[-1]['reused_pixels'] > 0, captures
sequence = captures.copy()

if hasattr(scene.render.image_settings, 'media_type'):
    scene.render.image_settings.media_type = 'VIDEO'
else:
    scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.filepath = str(out / 'camera.mp4')
captures.clear()
bpy.ops.render.render(animation=True)
videos = list(out.glob('camera*.mp4'))
assert videos and videos[0].stat().st_size > 1000, videos
assert len(captures) == 3 and captures[-1]['reused_pixels'] > 0, captures
(out / 'results.json').write_text(json.dumps(dict(sequence=sequence, video=captures), indent=2))
if hasattr(scene.render.image_settings, 'media_type'):
    scene.render.image_settings.media_type = 'IMAGE'
scene.render.image_settings.file_format = 'PNG'
for denoiser in ('OPTIX', 'OPENIMAGEDENOISE', 'DLSS'):
    settings.denoise = True
    settings.denoiser = denoiser
    settings.samples = 1 if denoiser == 'DLSS' else 4
    scene.render.filepath = str(out / f'still_{denoiser}.png')
    bpy.ops.render.render(write_still=True)
    assert (out / f'still_{denoiser}.png').exists()
    assert captures[-1]['gpu_updates'] >= 3, captures[-1]
    assert captures[-1]['reused_pixels'] > 0, captures[-1]

settings.render_mode = 'REFERENCE'
settings.samples = 4
settings.denoise = False
scene.render.filepath = str(out / 'reference.png')
bpy.ops.render.render(write_still=True)
assert captures[-1]['stored_frames'] == 0
settings.render_mode = 'REALTIME'
settings.denoise = True
settings.denoiser = 'OPTIX'
scene.render.filepath = str(out / 'camera_')
native_engine.render = original
if '--reject-border' in sys.argv:
    scene.render.use_border = True
    bpy.ops.render.render()  # Expected nonzero exit with a specific unsupported-setting error.
    raise AssertionError('RealTime render border must be rejected')
bpy.ops.photara_rs.reset_history()
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'realtime_camera_demo.blend'))
print('BLENDER_REALTIME_OUTPUT_PASS', flush=True)
