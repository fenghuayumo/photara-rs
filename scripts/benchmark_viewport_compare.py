"""Compare rendered camera feedback, not UI redraw rate, in an existing blend.

blender --factory-startup --window-geometry 0 0 1280 800 scene.blend
    --python scripts/benchmark_viewport_compare.py -- --engine PHOTARA_RS
    --materials neutral --denoiser OPTIX --output test_outputs/sponza.json

Only the disposable Blender process is changed; the input blend is never saved.
A small emissive near-camera probe encodes camera translation in its screen X.
Reading a three-pixel stripe after view_draw measures the displayed camera pose.
Noise/refinement redraws do not count as new motion frames. Latency is recovered
from the timestamped, projected input trajectory, excluding sweep reversals.
This is software input-to-framebuffer latency, excluding monitor/OS latency.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import addon_utils
import bpy
import gpu
import numpy as np
from mathutils import Vector
from bpy_extras.view3d_utils import location_3d_to_region_2d


parser = argparse.ArgumentParser()
parser.add_argument('--engine', choices=('PHOTARA_RS', 'CYCLES'), required=True)
parser.add_argument('--materials', choices=('original', 'neutral'), default='neutral')
parser.add_argument('--denoiser', choices=('NONE', 'OPTIX', 'DLSS'), default='OPTIX')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--upscale', choices=('NONE','QUALITY','BALANCED','PERFORMANCE','ULTRA_PERFORMANCE'), default='NONE')
parser.add_argument('--sweep-seconds', type=float, default=3.0)
parser.add_argument('--poll-ms', type=float)
parser.add_argument('--restart-ms', type=float)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
os.environ['CYCLES_VIEWPORT_FULL_RESOLUTION'] = '1'
addon_utils.enable('photara_rs_engine')
from photara_rs_engine import engine as photara_engine
import cycles
import photara_rs
if args.poll_ms is not None:
    photara_engine.ViewportState.POLL_INTERVAL = args.poll_ms/1000
if args.restart_ms is not None:
    photara_engine.ViewportState.MIN_RESTART_INTERVAL = args.restart_ms/1000

scene = bpy.context.scene
metadata = dict(blend=bpy.data.filepath, objects=len(scene.objects),
                polygons=sum(len(m.polygons) for m in bpy.data.meshes),
                materials=args.materials, denoiser=args.denoiser, engine=args.engine, upscale=args.upscale,
                blender=bpy.app.version_string, samples=4096, max_bounces=8,
                poll_ms=photara_engine.ViewportState.POLL_INTERVAL*1000,
                min_restart_ms=photara_engine.ViewportState.MIN_RESTART_INTERVAL*1000,
                navigation_poll_ms=getattr(photara_engine.ViewportState, 'NAVIGATION_POLL_INTERVAL',
                                           photara_engine.ViewportState.POLL_INTERVAL)*1000,
                addon_path=photara_engine.__file__,
                backend_path=('_photara_native: embedded Photara core'
                              if args.engine == 'PHOTARA_RS' and __import__('photara_rs_engine.native_backend', fromlist=['available']).available()
                              else photara_rs._backend._core.__file__),
                queued_camera_api=hasattr(photara_rs._backend._core.Context, 'queue_camera_matrix'))
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
for device in prefs.devices:
    device.use = device.type == 'OPTIX'
metadata['devices'] = [d.name for d in prefs.devices if d.use]
scene.cycles.device = 'GPU'
scene.cycles.preview_samples = 4096
scene.cycles.use_preview_adaptive_sampling = False
scene.cycles.max_bounces = 8
scene.cycles.diffuse_bounces = 8
scene.cycles.glossy_bounces = 8
scene.cycles.transmission_bounces = 8
scene.cycles.preview_denoising_upscale_quality = args.upscale
scene.cycles.preview_denoising_quality = 'HIGH'
scene.cycles.use_pixel_jitter = True
scene.cycles.pixel_filter_type = 'BLACKMAN_HARRIS'
scene.cycles.filter_width = 1.5
scene.render.preview_pixel_size = '1'
scene.camera.data.dof.use_dof = False
scene.cycles.use_preview_denoising = args.denoiser != 'NONE'
if args.denoiser != 'NONE':
    scene.cycles.preview_denoiser = args.denoiser
scene.cycles.preview_denoising_start_sample = 1
scene.cycles.preview_denoising_input_passes = 'RGB_ALBEDO_NORMAL'
scene.cycles.preview_denoising_prefilter = 'FAST'
scene.photara.viewport_samples = 4096
scene.photara.viewport_adaptive_sampling = False
scene.photara.max_bounces = 8
scene.photara.viewport_denoise = args.denoiser != 'NONE'
scene.photara.viewport_denoise_start_sample = 1
scene.photara.viewport_denoise_prefilter = 'FAST'
scene.photara.viewport_denoise_quality = 'HIGH'
scene.photara.viewport_denoise_passes = 'RGB_ALBEDO_NORMAL'
if args.denoiser != 'NONE':
    scene.photara.viewport_denoiser = args.denoiser
scene.photara.viewport_denoise_upscale_quality = args.upscale
scene.render.film_transparent = False
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'
scene.view_settings.exposure = 0
scene.view_settings.gamma = 1

if args.materials == 'neutral':
    material = bpy.data.materials.new('Benchmark neutral')
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (.5, .5, .5, 1)
    bsdf.inputs['Roughness'].default_value = .5
    for mesh in bpy.data.meshes:
        mesh.materials.clear()
        mesh.materials.append(material)
        for polygon in mesh.polygons:
            polygon.material_index = 0
    world = bpy.data.worlds.new('Benchmark world')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (.05, .05, .05, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 1
    scene.world = world

area = max((a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'),
           key=lambda a: a.width * a.height)
with bpy.context.temp_override(area=area):
    bpy.ops.screen.screen_full_area(use_hide_panels=True)
area = max((a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'),
           key=lambda a: a.width*a.height)
space = area.spaces.active
space.show_region_ui = False
space.show_region_toolbar = False
space.overlay.show_overlays = False
space.show_gizmo = False
space.shading.use_scene_world_render = True
space.shading.use_scene_lights_render = True
view = space.region_3d
view.view_perspective = 'PERSP'
camera = scene.camera
assert camera, 'Scene needs a camera for a reproducible starting view'
view.view_rotation = camera.matrix_world.to_quaternion()
space.lens = camera.data.lens
view.view_distance = 10
view.view_location = camera.location + view.view_rotation @ Vector((0, 0, -10))
base_location = view.view_location.copy()
right = view.view_rotation @ Vector((1, 0, 0))
up = view.view_rotation @ Vector((0, 1, 0))
forward = view.view_rotation @ Vector((0, 0, -1))
bpy.context.preferences.view.show_splash = False

state = dict(start=time.perf_counter(), phase='setup', inputs=[], frames=[],
             latencies=[], read_ms=[], callback_ms=[], input_gaps=[],
             callback=None, probe=None, offset=None, last_x=None,
             errors=[], restarts=0, queued_cameras=0, sync_ms=[], update_ms=[], native={}, trace_sizes={})


def fail(exc):
    traceback.print_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(error=str(exc), **metadata), indent=2))
    os._exit(1)


def project(context):
    return location_3d_to_region_2d(context.region, context.region_data, state['probe'])


def sample_frame(context):
    if state['probe'] is None or context.area != area:
        return
    point = project(context)
    row = int(round(point.y))
    begin = time.perf_counter()
    pixels = np.asarray(gpu.state.active_framebuffer_get().read_color(
        0, max(0, row - 1), context.region.width, 3, 4, 0, 'UBYTE'),
        dtype=np.uint8).reshape(3, context.region.width, 4)
    now = time.perf_counter()
    state['read_ms'].append((now - begin) * 1000)
    rgb = pixels[..., :3].astype(np.int16)
    mask = (rgb[..., 0] > 150) & (rgb[..., 2] > 150) & (rgb[..., 1] < 80)
    columns = np.flatnonzero(mask.sum(axis=0) >= 2)
    if len(columns) < 4:
        return
    runs = np.split(columns, np.where(np.diff(columns) > 1)[0] + 1)
    runs = [run for run in runs if 4 <= len(run) < 100]
    if not runs:
        return
    observed = float(np.mean(max(runs, key=len)))
    if state['phase'] == 'warmup':
        state['offset'] = observed - point.x
        state['last_probe_seen'] = now
        return
    if state['phase'] != 'motion' or state['offset'] is None:
        return
    previous = state['last_x']
    # Pixel-center quantization; require movement rather than sampling noise.
    if previous is not None and abs(observed - previous) < 1.5:
        return
    state['last_x'] = observed
    elapsed = now - state['motion_start']
    state['frames'].append(dict(time=now, x=observed, elapsed=elapsed))
    sweep = args.sweep_seconds
    if elapsed % sweep < .25 or elapsed % sweep > sweep - .25:
        return
    target = observed - state['offset']
    history = state['inputs']
    # Interpolate the most recent same-direction input crossing. Ignore the
    # reversal window, where an old frame could match the opposite sweep.
    for first, second in zip(reversed(history[:-1]), reversed(history[1:])):
        if now - first['time'] > 1:
            break
        if first['sweep'] != int(elapsed / sweep) or second['sweep'] != int(elapsed / sweep):
            continue
        x0, x1 = first['x'], second['x']
        if x0 == x1 or not min(x0, x1) <= target <= max(x0, x1):
            continue
        input_time = first['time'] + (second['time'] - first['time']) * (target - x0) / (x1 - x0)
        state['latencies'].append((now - input_time) * 1000)
        break


def wrap_draw(original):
    def draw(self, context, depsgraph):
        try:
            begin = time.perf_counter()
            result = original(self, context, depsgraph)
            if state['phase'] == 'motion':
                state['callback_ms'].append((time.perf_counter() - begin) * 1000)
            if args.engine == 'PHOTARA_RS' and self._viewport:
                import _cycles
                state['native'] = _cycles.photara_display_stats(self._viewport._display_capsule)
            elif args.engine == 'PHOTARA_RS' and getattr(self, '_native', False) and self.session:
                import _photara_native
                native = _photara_native.session_stats(self.session)
                if native:
                    state['native'] = native
                    if state['phase'] == 'motion':
                        size = f"{native['trace_width']}x{native['trace_height']}"
                        state['trace_sizes'][size] = state['trace_sizes'].get(size, 0)+1
            if args.engine == 'CYCLES' and self.session:
                import _cycles
                if hasattr(_cycles, 'session_stats'):
                    native = _cycles.session_stats(self.session)
                    if native:
                        state['native'] = native
                        if state['phase'] == 'motion':
                            size = f"{native['trace_width']}x{native['trace_height']}"
                            state['trace_sizes'][size] = state['trace_sizes'].get(size, 0)+1
            sample_frame(context)
            state['context'] = context.copy()
            return result
        except Exception as exc:
            fail(exc)
    return draw


engine_class = photara_engine.PhotaraRenderEngine if args.engine == 'PHOTARA_RS' else cycles.CyclesRender
engine_class.view_draw = wrap_draw(engine_class.view_draw)
if args.engine == 'PHOTARA_RS':
    photara_engine.PhotaraRenderEngine._report_viewport_failure = lambda self, exc: state['errors'].append(str(exc))
    original_restart = photara_engine.ViewportState.restart_render
    def restart(self):
        if state['phase'] == 'motion':
            state['restarts'] += 1
        return original_restart(self)
    photara_engine.ViewportState.restart_render = restart
    original_camera_sync = photara_engine.sync.SceneSyncState.sync_camera_view
    def camera_sync(self, *a, **kw):
        if state['phase'] == 'motion' and kw.get('queued') is not None:
            state['queued_cameras'] += 1
        return original_camera_sync(self, *a, **kw)
    photara_engine.sync.SceneSyncState.sync_camera_view = camera_sync
    original_sync = photara_engine.ViewportState.apply_updates
    def apply_updates(self, context, depsgraph):
        begin = time.perf_counter()
        result = original_sync(self, context, depsgraph)
        if state['phase'] == 'motion':
            state['sync_ms'].append((time.perf_counter() - begin) * 1000)
        return result
    photara_engine.ViewportState.apply_updates = apply_updates


def summarize(values):
    values = sorted(values)
    return dict(count=len(values), median=statistics.median(values) if values else None,
                p95=values[int((len(values)-1)*.95)] if values else None,
                max=max(values) if values else None)


def finish():
    frames = state['frames']
    intervals = [(b['time']-a['time'])*1000 for a,b in zip(frames, frames[1:])]
    result = dict(**metadata, viewport=state['size'], duration_seconds=args.sweep_seconds*4,
                  motion_frames=len(frames), motion_fps=len(frames)/(args.sweep_seconds*4),
                  motion_frame_interval_ms=summarize(intervals),
                  input_to_visible_pose_ms=summarize(state['latencies']),
                  framebuffer_read_ms=summarize(state['read_ms']),
                  view_draw_ms=summarize(state['callback_ms']),
                  apply_updates_ms=summarize(state['sync_ms']),
                  input_gap_ms=summarize(state['input_gaps']), native=state['native'],
                  render_restarts=state['restarts'], queued_camera_updates=state['queued_cameras'],
                  trace_size_observations=state['trace_sizes'],
                  synchronization_backend=('Cycles BlenderSync' if args.engine=='CYCLES' else
                                           'native BlenderSync' if state['native'].get('shader_nodes') else 'legacy'),
                  projection_precision_ms=1000*args.sweep_seconds/(.6*state['size'][0]),
                  raw_frames=frames, raw_inputs=state['inputs'])
    assert len(frames)>5 and state['latencies'], 'Probe did not track rendered motion'
    if state['trace_sizes']:
        factor = {'NONE':1, 'QUALITY':1.5, 'BALANCED':1/.58,
                  'PERFORMANCE':2, 'ULTRA_PERFORMANCE':3}[args.upscale]
        if args.denoiser != 'DLSS':
            factor = 1
        expected = f"{int(state['size'][0]/factor)}x{int(state['size'][1]/factor)}"
        assert set(state['trace_sizes']) == {expected}, state['trace_sizes']
        assert not state['native']['use_resolution_divider'], state['native']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print('VIEWPORT_COMPARE',json.dumps({k:v for k,v in result.items() if not k.startswith('raw_')}),flush=True)
    bpy.ops.wm.quit_blender()


def tick():
    try:
        now = time.perf_counter()
        assert not state['errors'], state['errors']
        assert now-state['start']<180, 'Viewport benchmark timed out'
        region = next(r for r in area.regions if r.type == 'WINDOW')
        if state['phase'] == 'setup':
            with bpy.context.temp_override(area=area, region=region):
                bpy.context.view_layer.update()
                # Force projection to reflect the requested lens/view.
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                depth = 1.0
                center = camera.location + forward*depth - up*(.45*depth/view.window_matrix[1][1])
                half_width = depth/view.window_matrix[0][0]
                state['half_width'] = half_width
                state['probe'] = center
                state['size'] = [region.width, region.height]
                bpy.ops.mesh.primitive_plane_add(size=1, location=center)
                obj = bpy.context.object
                obj.name = 'Viewport latency probe'
                obj.rotation_euler = view.view_rotation.to_euler()
                obj.scale = (.035*half_width, .08*half_width, 1)
                mat = bpy.data.materials.new('Viewport latency magenta')
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes['Principled BSDF']
                bsdf.inputs['Base Color'].default_value = (0,0,0,1)
                bsdf.inputs['Emission Color'].default_value = (1,0,1,1)
                bsdf.inputs['Emission Strength'].default_value = 1
                obj.data.materials.append(mat)
                view.view_location = base_location-right*(.6*half_width)
                scene.render.engine = args.engine
                space.shading.type = 'RENDERED'
                state['phase'] = 'warmup'
                state['warmup_start'] = now
            return .02
        if state['phase'] == 'warmup':
            if now-state['warmup_start']<5 or now-state.get('last_probe_seen',0)>.5:
                area.tag_redraw()
                return .02
            state['phase'] = 'motion'
            state['motion_start'] = now
            state['last_input'] = now
            bpy.ops.screen.screenshot(filepath=str(args.output.with_suffix('.png').resolve()))
            print('MOTION_START', args.engine, state['size'], flush=True)
        elapsed = now-state['motion_start']
        if elapsed>=args.sweep_seconds*4:
            finish()
            return None
        phase = elapsed/args.sweep_seconds
        fraction = phase % 1
        position = (-.6+1.2*fraction) if int(phase)%2==0 else (.6-1.2*fraction)
        view.view_location = base_location+right*(position*state['half_width'])
        # view_matrix is updated during Blender's next draw, so projecting it
        # here would timestamp the previous input. Fixed orientation makes the
        # new projection analytic: screen X = W/2 - translation*W/(2*half_width).
        expected_x = state['size'][0] * .5 * (1-position)
        state['inputs'].append(dict(time=now, x=expected_x, sweep=int(phase)))
        state['input_gaps'].append((now-state['last_input'])*1000)
        state['last_input'] = now
        area.tag_redraw()
        return 1/120
    except Exception as exc:
        fail(exc)


bpy.app.timers.register(tick, first_interval=.2)
