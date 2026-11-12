"""Run with Blender opening assets_room.blend; verify the installed viewport."""
import bpy,gpu,numpy as np,time,traceback,os
from pathlib import Path
from photara_rs_engine import engine
root=Path(__file__).resolve().parent.parent
scene=bpy.context.scene
scene.render.engine='PHOTARA_RS'
scene.photara_rs.viewport_samples=4
scene.photara_rs.viewport_denoise=True
scene.photara_rs.viewport_denoiser='DLSS'
scene.photara_rs.viewport_denoise_upscale_quality='NONE'
bpy.context.preferences.view.show_splash=False
area=next(a for a in bpy.context.screen.areas if a.type=='VIEW_3D')
errors=[];checked=False
engine.PhotaraRenderEngine._report_viewport_failure=lambda self,exc:errors.append(str(exc))
original=engine.ViewportState.draw
def draw(self,context):
 global checked
 original(self,context)
 if checked or self.needs_update or self.job is not None:return
 material=bpy.data.materials['Natural_pomegranate_peel']
 exported=self.sync_state.materials[material.as_pointer()]
 assert exported.base_color_texture is not None
 import _cycles
 stats=_cycles.photara_display_stats(self._display_capsule)
 assert stats['gpu_updates']>0,stats
 pixels=np.asarray(gpu.state.active_framebuffer_get().read_color(0,0,self.width,self.height,4,0,'FLOAT'),dtype=np.float32).reshape(-1,4)
 assert np.isfinite(pixels).all()
 assert np.std(pixels[:,:3])>.01,'Blank viewport'
 checked=True
 print('FULL_ROOM_VIEWPORT_PASS',exported.base_color_texture,stats,flush=True)
engine.ViewportState.draw=draw
started=time.monotonic()
def tick():
 try:
  assert not errors,errors
  if checked:
   bpy.ops.screen.screenshot(filepath=str(root/'test_outputs/room_viewport_fixed.png'))
   bpy.ops.wm.quit_blender();return None
  assert time.monotonic()-started<90,'Room viewport timeout'
  area.tag_redraw();return .1
 except Exception:
  traceback.print_exc();os._exit(1)
bpy.app.timers.register(tick,first_interval=.1)
area.spaces.active.shading.type='RENDERED'
