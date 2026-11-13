"""Run in Blender background with a loaded blend; never saves the input file.

Measures the standalone REALTIME API through the legacy Blender converter.
Includes synchronous render completion and CPU beauty readback, excludes display.
"""
import json
from pathlib import Path
import statistics
import sys
import time

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'photara/python'))
import photara_rs as pr
from photara_rs_engine.sync import SceneSyncState

out = ROOT / 'test_outputs/sponza_realtime'
out.mkdir(parents=True, exist_ok=True)
results = {'source': bpy.data.filepath, 'backend': pr._backend._core.__file__,
           'blender': bpy.app.version_string, 'objects': len(bpy.context.scene.objects),
           'source_polygons': sum(len(m.polygons) for m in bpy.data.meshes),
           'conversion': 'legacy SceneSyncState (limited material conversion)',
           'measurement': 'render()+beauty readback; excludes display; 8 warmup, 60 measured frames',
           'max_bounces': 8, 'samples_per_frame': 1, 'runs': []}

def stats(values):
    return {'mean_ms': statistics.mean(values), 'median_ms': statistics.median(values),
            'p95_ms': float(np.percentile(values, 95)), 'min_ms': min(values),
            'max_ms': max(values), 'fps': 1000 / statistics.mean(values)}

for width, height in [(1280, 720), (1920, 1080)]:
    for denoise in [False, True]:
        start = time.perf_counter()
        with pr.Scene(width, height, samples=1, max_bounces=8, adaptive_sampling=False,
                      denoise=denoise, denoiser='OPTIX', render_mode='REALTIME',
                      samples_per_frame=1) as scene:
            sync = SceneSyncState(scene)
            sync.sync(bpy.context.evaluated_depsgraph_get())
            camera = bpy.context.scene.camera
            sync.sync_camera(camera, width / height)
            setup_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            scene.render()
            first_ms = (time.perf_counter() - start) * 1000
            base = camera.location.copy()
            for moving in [False, True]:
                times = []
                for i in range(68):
                    start = time.perf_counter()
                    if moving:
                        camera.location = base + camera.matrix_world.to_quaternion() @ __import__('mathutils').Vector((0.25 * np.sin(i * .07), 0, 0))
                        bpy.context.view_layer.update()
                        sync.sync_camera(camera, width / height)
                    frame = scene.render()
                    pixels = frame.beauty
                    elapsed = (time.perf_counter() - start) * 1000
                    if not np.isfinite(pixels).all():
                        raise RuntimeError('Non-finite image')
                    if i >= 8:
                        times.append(elapsed)
                if not moving:
                    rgb = np.maximum(pixels[..., :3], 0)
                    rgb = rgb / (1 + rgb)
                    rgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)
                    image = bpy.data.images.new('benchmark', width=width, height=height, alpha=True)
                    rgba = np.concatenate([np.clip(rgb, 0, 1), np.ones((height, width, 1))], axis=2)
                    image.pixels.foreach_set(np.flipud(rgba).astype(np.float32).ravel())
                    image.filepath_raw = str(out / f'{width}x{height}_{"optix" if denoise else "none"}.png')
                    image.file_format = 'PNG'
                    image.save()
                    bpy.data.images.remove(image)
                run = dict(width=width, height=height, denoiser='OPTIX' if denoise else 'NONE',
                           moving_camera=moving, setup_ms=setup_ms, first_frame_ms=first_ms,
                           frames=60, frame_ms=times, **stats(times))
                results['runs'].append(run)
                (out / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
                print('REALTIME_BENCH', json.dumps({k: v for k, v in run.items() if k != 'frame_ms'}), flush=True)
            camera.location = base
            bpy.context.view_layer.update()
            sync.close()
