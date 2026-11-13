"""GPU/host feedback parity, 1-SPP candidate energy and Reference isolation."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
from test_neeat import build_scene
import photara_rs as pr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--open-lights', action='store_true')
args = parser.parse_args()

def open_scene(size, samples):
    scene = pr.Scene(size, size, samples=samples, max_bounces=3,
                     denoise=False, adaptive_sampling=False)
    scene.camera.location = (0, 2.4, -4)
    scene.camera.target = (0, 0, 1)
    scene.world.color = (0, 0, 0)
    metal = scene.materials.new('metal', base_color=(.7, .7, .7), roughness=.25, metallic=1)
    mesh = scene.meshes.new('floor', [(-3, 0, -1), (-3, 0, 5), (3, 0, 5), (3, 0, -1)],
                            [(0, 1, 2), (0, 2, 3)], material=metal)
    scene.objects.new('floor', mesh)
    for i in range(32):
        angle = 2 * np.pi * i / 32
        scene.lights.new_area(f'light{i}', position=(2*np.sin(angle), 2, 2*np.cos(angle)+1),
                              target=(0, 0, 1), size=(.2, .2), strength=20)
    return scene

if args.open_lights: build_scene = open_scene

out = Path('test_outputs/nee_gpu_reservoir_open' if args.open_lights else 'test_outputs/nee_gpu_reservoir')
out.mkdir(parents=True, exist_ok=True)
results = []
reference = None
arrays = {}
for backend in ('CPU', 'GPU'):
    if backend == 'CPU': os.environ['PHOTARA_NEEAT_CPU'] = '1'
    else: os.environ.pop('PHOTARA_NEEAT_CPU', None)
    for candidates in (1, 4, 8):
        if backend == 'CPU' and candidates != 1: continue
        with build_scene(64, 4096) as scene:
            if reference is None:
                reference = scene.render().rgb.copy()
                arrays['reference'] = reference
            scene.configure_nee_candidates(candidates)
            scene.configure_render_mode('REALTIME', samples_per_frame=1)
            for _ in range(32): scene.render()
            images, times = [], []
            for i in range(256):
                start = time.perf_counter()
                image = scene.render().rgb.copy()
                times.append((time.perf_counter() - start) * 1000)
                assert np.isfinite(image).all()
                images.append(image)
            images = np.asarray(images)
            mean = images.mean(axis=0)
            result = dict(backend=backend, candidates=candidates,
                          median_ms=statistics.median(times), mean_ms=statistics.mean(times),
                          mean_luminance=float(mean.mean()), reference_luminance=float(reference.mean()),
                          mean_bias=float((mean.mean() - reference.mean()) / reference.mean()),
                          mean_frame_mse=float(np.mean((images-reference)**2)),
                          mean_rmse=float(np.sqrt(np.mean((mean-reference)**2))), stats=scene.neeat_stats())
            assert abs(result['mean_bias']) < .08, result
            assert result['stats']['gpu_updates' if backend == 'GPU' else 'cpu_updates'] == 288
            assert result['stats']['cpu_updates' if backend == 'GPU' else 'gpu_updates'] == 0
            scene.configure_render_mode('REFERENCE')
            repeat = scene.render().rgb.copy()
            np.testing.assert_allclose(reference, repeat, rtol=1e-5, atol=1e-6,
                                       err_msg='Reference changed with candidate count/backend')
            arrays[f'{backend}_{candidates}'] = mean
            results.append(result)
            (out / 'summary.json').write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
np.savez_compressed(out / 'images.npz', **arrays)
cpu, gpu = results[:2]
assert abs(cpu['mean_luminance'] - gpu['mean_luminance']) / reference.mean() < .03
assert abs(cpu['stats']['max_pdf'] - gpu['stats']['max_pdf']) < .03
print('GPU/CPU parity, finite output, energy and Reference isolation passed.', flush=True)
