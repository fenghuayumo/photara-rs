"""Run Blender --factory-startup --python this script, optionally -- --viewport."""
import sys,time,os,traceback
from pathlib import Path
import bpy,numpy as np,addon_utils
ROOT=Path(__file__).resolve().parent.parent
if not (ROOT/'test_outputs/gaussians_fixture.ply').is_file():
 sys.path.insert(0,str(ROOT/'photara/python/tests'))
 from test_gsplat import GSplatTests
 import shutil
 fixture=GSplatTests();fixture.setUp()
 shutil.copyfile(fixture.fixture('binary_little_endian'),ROOT/'test_outputs/gaussians_fixture.ply')
 fixture.tearDown()
addon_utils.disable('photara_rs_engine')
for name in list(sys.modules):
 if name=='photara_rs_engine' or name.startswith('photara_rs_engine.'):del sys.modules[name]
sys.path[:0]=[str(ROOT),str(ROOT/'photara/python')]
import photara_rs_engine as addon
addon.register()
from photara_rs_engine import sync,engine
import photara_rs as pr
for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
assert bpy.ops.import_scene.photara_gaussians(filepath=str(ROOT/'test_outputs/gaussians_fixture.ply')) == {'FINISHED'}
cloud=bpy.context.object
assert cloud.type=='POINTCLOUD',cloud.type
print('GSPLAT_ATTRIBUTES',[(a.name,a.data_type) for a in cloud.data.attributes],flush=True)
for a in cloud.data.attributes:
 if a.name in ('rotation','radiance:base'):print('ROTATION_RNA',[p.identifier for p in a.data[0].bl_rna.properties],flush=True)
data=sync.gaussian_pointcloud_data(cloud.data)
assert data['sh_rest'].shape==(2,15,3)
scene=bpy.context.scene
with pr.Scene(32,32,samples=1,denoise=False) as native:
 state=sync.SceneSyncState(native)
 state.sync(bpy.context.evaluated_depsgraph_get())
 assert len(state._gaussian_objects)==1
 cloud.location.x+=.1;bpy.context.view_layer.update()
 state.sync(bpy.context.evaluated_depsgraph_get())
bpy.ops.object.camera_add(location=(0,0,-3))
camera=bpy.context.object;camera.rotation_euler=(cloud.location-camera.location).to_track_quat('-Z','Y').to_euler();scene.camera=camera
scene.render.engine='PHOTARA_RS';scene.photara.samples=8;scene.photara.denoise=False;scene.photara.adaptive_sampling=False
scene.render.resolution_x=128;scene.render.resolution_y=96;scene.render.resolution_percentage=100
scene.world.color=(0,0,0);scene.world.use_nodes=False
frames=[];old=engine.PhotaraRenderEngine._fill_render_result
def capture(self,result,frame,layer):
 old(self,result,frame,layer);frames.append(frame.beauty.copy())
engine.PhotaraRenderEngine._fill_render_result=capture
scene.render.filepath=str(ROOT/'test_outputs/blender_gsplat_final.png')
bpy.ops.render.render(write_still=True)
assert len(frames)==1 and np.isfinite(frames[0]).all()
rgb=frames[0][...,:3]
assert ((rgb[...,0]>.1)&(rgb[...,2]<.05)).sum()>20
assert ((rgb[...,2]>.1)&(rgb[...,0]<.05)).sum()>20
print('BLENDER_GSPLAT_FINAL_PASS',flush=True)
if '--viewport' in sys.argv:
 import gpu
 scene.photara.viewport_samples=16;scene.photara.viewport_denoiser='DLSS'
 area=next(a for a in bpy.context.screen.areas if a.type=='VIEW_3D')
 area.spaces.active.region_3d.view_perspective='CAMERA'
 area.spaces.active.shading.use_scene_world_render=True
 errors=[];ready=False;draw=engine.ViewportState.draw
 engine.PhotaraRenderEngine._report_viewport_failure=lambda self,exc:errors.append(str(exc))
 def checked(self,context):
  global ready
  draw(self,context)
  if self.needs_update or self.job is not None:return
  values=np.asarray(gpu.state.active_framebuffer_get().read_color(0,0,self.width,self.height,4,0,'FLOAT')).reshape(-1,4)
  assert ((values[:,0]>.1)&(values[:,2]<.05)).sum()>20
  assert len(self.sync_state._gaussian_objects)==1
  ready=True
 engine.ViewportState.draw=checked
 started=time.monotonic()
 def tick():
  try:
   assert not errors,errors
   assert time.monotonic()-started<60,'Viewport timeout'
   if ready:
    bpy.ops.screen.screenshot(filepath=str(ROOT/'test_outputs/blender_gsplat_viewport.png'))
    print('BLENDER_GSPLAT_VIEWPORT_PASS',flush=True);bpy.ops.wm.quit_blender();return None
   area.tag_redraw();return .05
  except Exception:traceback.print_exc();os._exit(1)
 bpy.app.timers.register(tick,first_interval=.1)
 area.spaces.active.shading.type='RENDERED'
