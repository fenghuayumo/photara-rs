"""Open a .blend in background Blender, then compare native NEE-AT on/off.

Writes outputs only; never saves or modifies the source blend on disk. Both
low-SPP cases use RealTime with equal warm-up and no history or denoising.
"""
import argparse
import collections
import json
from pathlib import Path
import sys
import time

import addon_utils
import bpy
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=360)
parser.add_argument('--samples', type=int, default=4)
parser.add_argument('--reference-samples', type=int, default=256)
parser.add_argument('--warmup', type=int, default=8)
parser.add_argument('--repeats', type=int, default=1)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
assert min(args.width, args.height, args.samples, args.reference_samples) > 0
assert args.warmup >= 0 and args.repeats > 0
args.output.mkdir(parents=True, exist_ok=True)
addon_utils.enable('photara_rs_engine')
from photara_rs_engine import native_backend, native_engine
import _photara_native

assert native_backend.available(), 'Requires the embedded Photara renderer'
# Compare the same pixel footprints. Per-frame camera jitter is useful for
# reconstruction, but without reconstruction it confounds pixelwise MSE.
configure = native_backend._configure


def configure_without_frame_jitter(scene):
    configure(scene)
    scene.photara_native.use_pixel_jitter = False


native_backend._configure = configure_without_frame_jitter

scene = bpy.context.scene
assert scene.camera and scene.camera.data.type == 'PERSP'
scene.render.engine = 'PHOTARA_RS'
scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
scene.render.resolution_percentage = 100
scene.render.use_border = False
scene.render.use_motion_blur = False
scene.render.use_multiview = False
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_depth = '32'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.use_persistent_data = True
p = scene.photara_rs
p.denoise = False
p.adaptive_sampling = False
p.history_samples = 0
p.realtime_warmup_frames = args.warmup
p.seed = 42
reports = {}
timings = []
render = native_engine.render


def capture(engine, depsgraph):
    start = time.perf_counter()
    render(engine, depsgraph)
    timings.append(dict(render_ms=(time.perf_counter() - start) * 1000,
                        session=dict(_photara_native.session_stats(engine.session)),
                        history=dict(_photara_native.realtime_stats(engine.session))))


native_engine.render = capture
metadata = dict(source=bpy.data.filepath, camera=scene.camera.name,
                lights=dict(collections.Counter(o.data.type for o in scene.objects
                                                if o.type == 'LIGHT')),
                width=args.width, height=args.height, samples=args.samples,
                reference_samples=args.reference_samples, warmup=args.warmup, repeats=args.repeats,
                denoise=False, history_frames=0, frame_pixel_jitter=False,
                timing_scope='native render including sync, warm-up and reconstruction; excludes file writes',
                limitations='Independent warmed stills; timings include training and are not steady-frame throughput')
arrays = {}
for name, mode, samples, neeat in (
        ('reference', 'REFERENCE', args.reference_samples, False),
        ('neeat_off', 'REALTIME', args.samples, False),
        ('neeat_on', 'REALTIME', args.samples, True),
        ('neeat_pure', 'REALTIME', args.samples, True)):
    images, runs = [], []
    for repeat in range(1 if name == 'reference' else args.repeats):
        print('NEEAT_CASE_START', name, repeat, flush=True)
        p.render_mode, p.samples, p.realtime_neeat = mode, samples, neeat
        p.realtime_neeat_mode = 'PURE' if name == 'neeat_pure' else 'HYBRID'
        p.seed = 42 + repeat
        scene.render.image_settings.color_depth = '32'
        scene.render.filepath = str(args.output / (name + '.exr'))
        bpy.ops.render.render(write_still=True)
        run = timings[-1]
        assert run['history']['stored_frames'] == 0
        image = bpy.data.images.load(scene.render.filepath, check_existing=False)
        pixels = np.empty(args.width * args.height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgb = pixels.reshape(args.height, args.width, 4)[..., :3].copy()
        assert np.isfinite(rgb).all(), name
        images.append(rgb)
        if name != 'reference':
            run['linear_rgb_mse'] = float(np.mean((rgb.astype(np.float64) - arrays['reference']) ** 2))
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = '16'
        image.save_render(str(args.output / (name + '.png')), scene=scene)
        scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.data.images.remove(image)
        runs.append(run)
        print('NEEAT_CASE_DONE', name, repeat, json.dumps(run), flush=True)
    arrays[name] = np.mean(images, axis=0)
    reports[name] = dict(runs[-1], runs=runs,
                         median_render_ms=float(np.median([r['render_ms'] for r in runs])))
    if name != 'reference':
        reports[name]['mean_frame_mse'] = float(np.mean([r['linear_rgb_mse'] for r in runs]))

for name in ('neeat_off', 'neeat_on', 'neeat_pure'):
    reports[name]['linear_rgb_mse'] = float(np.mean(
        (arrays[name].astype(np.float64) - arrays['reference']) ** 2))
comparison = dict(max_absolute_difference=float(np.max(np.abs(
    arrays['neeat_on'] - arrays['neeat_off']))),
    on_active=reports['neeat_on']['session']['neeat_emitters'] > 0)
if not comparison['on_active']:
    comparison['note'] = ('NEE-AT did not activate; image noise differences and timing '
                          'differences do not measure NEE-AT benefit')
np.savez_compressed(args.output / 'linear.npz', **arrays)
result = dict(metadata=metadata, cases=reports, comparison=comparison)
(args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print('NEEAT_BENCHMARK_RESULT', json.dumps(result), flush=True)
