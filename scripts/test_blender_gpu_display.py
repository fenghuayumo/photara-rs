"""Verify actual Blender GPU interop, frame alignment and absence of CPU pixel transfers.

blender.exe --factory-startup --python-exit-code 1 --python scripts/test_blender_gpu_display.py
"""
import json
import os
from pathlib import Path
import time
import traceback

import bpy
import gpu
import numpy as np
from mathutils import Quaternion, Vector
from bpy_extras.view3d_utils import location_3d_to_region_2d

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_blender import prepare_scene
from photara_rs_engine import engine

OUTPUT = Path(__file__).resolve().parent.parent / 'test_outputs'
OUTPUT.mkdir(exist_ok=True)
scene, cube, material = prepare_scene()
scene.photara.viewport_samples = 16
scene.photara.viewport_denoise = False
bpy.context.preferences.view.show_splash = False
area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
area.spaces.active.region_3d.view_perspective = 'PERSP'
reports, errors, seen = [], [], set()
original_draw = engine.ViewportState.draw


def checked_draw(self, context):
    original_draw(self, context)
    assert self.gpu_display, 'Native Blender display bridge is not active'
    if self.job is not None or self.needs_update:
        return
    signature = repr(self.camera_matrix)
    if signature in seen:
        return
    import _cycles
    stats = _cycles.photara_display_stats(self._display_capsule)
    assert stats['gpu_updates'] > 0, stats
    assert stats['cpu_maps'] == 0, stats
    assert self.scene._session._ctx.preview(0) is None, 'CPU render capture is still active'
    pixels = np.asarray(gpu.state.active_framebuffer_get().read_color(
        0, 0, self.width, self.height, 4, 0, 'FLOAT'), dtype=np.float32).reshape(self.height, self.width, 4)
    mask = (pixels[..., 0] > 0.05) & (pixels[..., 0] > pixels[..., 1] * 1.5) & (pixels[..., 0] > pixels[..., 2] * 1.5)
    y, x = np.where(mask)
    assert x.size > 100, 'GPU driver did not draw the red test cube'
    actual = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
    projected = [location_3d_to_region_2d(context.region, context.region_data,
                                         cube.matrix_world @ Vector(c)) for c in cube.bound_box]
    expected = [min(v.x for v in projected), min(v.y for v in projected),
                max(v.x for v in projected), max(v.y for v in projected)]
    error = max(abs(a - b) for a, b in zip(actual, expected))
    assert error < 4, (actual, expected, error)
    seen.add(signature)
    reports.append(dict(actual=actual, expected=expected, error=error, **stats))
    print('PHOTARA_GPU_FRAME', json.dumps(reports[-1]), flush=True)


engine.ViewportState.draw = checked_draw
engine.PhotaraRenderEngine._report_viewport_failure = lambda self, exc: errors.append(str(exc))
start = time.monotonic()
stage = 0


def tick():
    global stage
    try:
        assert not errors, errors
        if time.monotonic() - start > 90:
            raise AssertionError('GPU viewport timed out')
        if len(reports) > stage:
            if stage == 2:
                (OUTPUT / 'gpu_display_verification.json').write_text(json.dumps(reports, indent=2))
                bpy.ops.screen.screenshot(filepath=str(OUTPUT / 'gpu_display_verified.png'))
                print('PHOTARA_GPU_INTEROP_PASS', flush=True)
                bpy.ops.wm.quit_blender()
                return None
            stage += 1
            view = area.spaces.active.region_3d
            view.view_rotation = Quaternion((0, 0, 1), 0.5) @ view.view_rotation
            view.view_location += Vector((0.4, 0.3, 0.2))
            area.tag_redraw()
        return 0.05
    except Exception:
        traceback.print_exc()
        os._exit(1)


bpy.app.timers.register(tick, first_interval=0.1)
area.spaces.active.shading.type = 'RENDERED'
