"""Interactive Blender regression: switch modes while the rendered viewport draws."""
import traceback
import bpy
import addon_utils

addon_utils.enable('photara_rs_engine')
from photara_rs_engine import native_backend

scene = bpy.context.scene
scene.render.engine = 'PHOTARA_RS'
scene.photara_rs.viewport_samples = 4
scene.photara_rs.viewport_denoise = False
area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
area.spaces.active.shading.type = 'RENDERED'
state = dict(drawing=False, draws=0, updates=0, errors=[])
original_draw = native_backend.view_draw
original_update = native_backend.view_update
original_configure = native_backend.configure


def configure(*args, **kwargs):
    assert not state['drawing'], 'Draw callback attempted ID configuration'
    return original_configure(*args, **kwargs)


def draw(*args, **kwargs):
    state['drawing'] = True
    try:
        original_draw(*args, **kwargs)
        state['draws'] += 1
    except Exception:
        state['errors'].append(traceback.format_exc())
        raise
    finally:
        state['drawing'] = False


def update(*args, **kwargs):
    try:
        original_update(*args, **kwargs)
        state['updates'] += 1
    except Exception:
        state['errors'].append(traceback.format_exc())
        raise


native_backend.configure = configure
native_backend.view_draw = draw
native_backend.view_update = update


def switch_mode():
    scene.render.use_persistent_data = False
    scene.photara_rs.render_mode = 'REALTIME'
    assert scene.render.use_persistent_data
    scene.photara_rs.history_samples = 2
    area.tag_redraw()


def finish():
    assert state['draws'] > 2 and state['updates'] > 0, state
    assert not state['errors'], state
    print('BLENDER_REALTIME_VIEWPORT_PASS', state, flush=True)
    bpy.ops.wm.quit_blender()


bpy.app.timers.register(switch_mode, first_interval=3)
bpy.app.timers.register(finish, first_interval=9)
