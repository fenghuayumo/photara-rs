"""DLSS final image and two-frame animation regression in Blender."""
import sys
from pathlib import Path
from types import SimpleNamespace
import bpy
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'photara/python')]
from test_blender import prepare_scene
from photara_rs_engine import engine
scene, cube, material = prepare_scene()
settings = scene.photara_rs
settings.denoiser = 'DLSS'
settings.denoise = True
settings.samples = 4
settings.denoise_min_samples = 16
settings.adaptive_sampling = True
assert engine._final_sampling_options(settings) == dict(samples=16, adaptive_sampling=False)
settings.samples = 32
assert engine._final_sampling_options(settings)['samples'] == 32
settings.samples = 4
captured = []
original = engine.PhotaraRenderEngine._fill_render_result
def capture(self, result, frame, layer):
    original(self, result, frame, layer)
    assert np.isfinite(frame.beauty).all()
    assert (frame.object_id == 7).sum() > 100
    captured.append(frame.beauty.copy())
engine.PhotaraRenderEngine._fill_render_result = capture
scene.render.resolution_x = 96
scene.render.resolution_y = 72
scene.render.resolution_percentage = 100
out = ROOT / 'test_outputs'
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(out / 'dlss_final_16spp.png')
bpy.ops.render.render(write_still=True)
assert len(captured) == 1
cube.keyframe_insert(data_path='location', frame=1)
cube.location.x += 0.7
cube.keyframe_insert(data_path='location', frame=2)
scene.frame_start = 1
scene.frame_end = 2
scene.render.filepath = str(out / 'dlss_animation_')
bpy.ops.render.render(animation=True)
assert len(captured) == 3
assert not np.allclose(captured[1], captured[2])
settings.denoise = False
assert engine._final_sampling_options(settings) == dict(samples=4, adaptive_sampling=True)
print('DLSS_FINAL_IMAGE_ANIMATION_PASS', flush=True)
