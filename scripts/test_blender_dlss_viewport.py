"""Verify successful DLSS evaluations in a real Blender rendered viewport.

Run without --background in the rebuilt Blender.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if '--installed' not in sys.argv:
    sys.path[:0] = [str(ROOT), str(ROOT / 'photara/python')]
import addon_utils
import bpy
import _photara_native
addon_utils.enable('photara_rs_engine')
from photara_rs_engine import native_backend
assert native_backend.available()
scene = bpy.context.scene
scene.render.engine = 'PHOTARA_RS'
p = scene.photara_rs
p.viewport_samples = 16
area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
phases = [(False, 'NONE', 1.0), (True, 'NONE', 1.0),
          (True, 'BALANCED', 1 / .58), (True, 'PERFORMANCE', 2.0),
          (False, 'NONE', 1.0)]
reports = []
state = dict(index=0, stats=None, baseline=0, changed=time.monotonic())
original = native_backend.view_draw


def draw(engine, context, depsgraph):
    original(engine, context, depsgraph)
    if engine.session:
        stats = _photara_native.session_stats(engine.session)
        if stats:
            state['stats'] = stats


def apply_phase():
    enabled, quality, _ = phases[state['index']]
    p.viewport_denoise = enabled
    p.viewport_denoiser = 'DLSS'
    p.viewport_denoise_upscale_quality = quality
    state['baseline'] = (state['stats'] or {}).get('dlss_evaluations', 0)
    state['stats'] = None
    state['changed'] = time.monotonic()
    # Exercise navigation as well as property changes.
    area.spaces.active.region_3d.view_distance += .1
    area.tag_redraw()


native_backend.view_draw = draw
apply_phase()
area.spaces.active.shading.type = 'RENDERED'
started = time.monotonic()


def tick():
    try:
        assert time.monotonic() - started < 180, state
        stats = state['stats']
        enabled, quality, factor = phases[state['index']]
        if stats and time.monotonic() - state['changed'] > 1:
            matches = stats['use_denoise'] == enabled
            if enabled:
                matches = matches and stats['denoiser_type'] == 8
                matches = matches and abs(stats['denoiser_upscale_factor'] - factor) < 1e-5
                matches = matches and stats['dlss_evaluations'] > state['baseline'] + 1
            if matches:
                reports.append(dict(enabled=enabled, quality=quality, **stats))
                print('DLSS_VIEWPORT_PHASE', json.dumps(reports[-1]), flush=True)
                state['index'] += 1
                if state['index'] == len(phases):
                    (ROOT / 'test_outputs/dlss_viewport.json').write_text(json.dumps(reports, indent=2))
                    print('DLSS_VIEWPORT_PASS', flush=True)
                    bpy.ops.wm.quit_blender()
                    return None
                apply_phase()
        area.tag_redraw()
        return .1
    except Exception:
        traceback.print_exc()
        os._exit(1)


bpy.app.timers.register(tick, first_interval=.1)
