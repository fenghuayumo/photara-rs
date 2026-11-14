"""Run in windowed Blender: F12 must display pixels before rendering finishes."""
from pathlib import Path
import sys
import time
import traceback
import os

import bpy
import gpu
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from test_blender import prepare_scene
from photara_rs_engine import native_engine, native_backend

assert not bpy.app.background, 'Requires a window to test the F12 display driver'
assert native_backend.available(), 'Requires the embedded Photara renderer'
scene, cube, material = prepare_scene()
scene.render.resolution_x, scene.render.resolution_y = 640, 480
scene.photara_rs.samples = 8192
scene.photara_rs.denoise = True
scene.photara_rs.denoiser = 'OPTIX'
bpy.context.preferences.view.render_display_type = 'WINDOW'
state = dict(rendering=False, finished=False, draws=0, visible=False)
render = native_engine.render
draw = native_engine.draw


def checked_render(engine, depsgraph):
    state['rendering'] = True
    try:
        render(engine, depsgraph)
    finally:
        state['rendering'] = False
        state['finished'] = True


def checked_draw(engine, depsgraph, space_image):
    draw(engine, depsgraph, space_image)
    if not state['rendering']:
        return
    state['draws'] += 1
    x, y, width, height = gpu.state.viewport_get()
    if width <= 0 or height <= 0:
        return
    pixels = np.asarray(gpu.state.active_framebuffer_get().read_color(
        x, y, width, height, 4, 0, 'FLOAT'), dtype=np.float32).reshape(-1, 4)
    red = (pixels[:, 0] > .05) & (pixels[:, 0] > pixels[:, 1] * 1.5) & (
        pixels[:, 0] > pixels[:, 2] * 1.5)
    if np.count_nonzero(red) > 100:
        state['visible'] = True


native_engine.render = checked_render
native_engine.draw = checked_draw
started = time.monotonic()


def start():
    bpy.ops.render.render('INVOKE_DEFAULT')


def finish():
    try:
        assert time.monotonic() - started < 120, state
        if not state['finished'] or bpy.app.is_job_running('RENDER'):
            return .1
        assert state['draws'] > 0 and state['visible'], state
        print('BLENDER_PROGRESSIVE_RENDER_PASS', state, flush=True)
        bpy.ops.wm.quit_blender()
    except Exception:
        traceback.print_exc()
        os._exit(1)


bpy.app.timers.register(start, first_interval=1)
bpy.app.timers.register(finish, first_interval=2)
