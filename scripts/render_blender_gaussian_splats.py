"""Render a 3DGS PLY/.splat through the installed Photara Blender add-on.
Run blender --background --factory-startup --python this_script -- model.ply
    --camera camera.json --output render.png
"""
import argparse,json,sys,math
from pathlib import Path
import bpy,addon_utils,numpy as np
from mathutils import Matrix,Vector
parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('--camera',required=True);parser.add_argument('--output',required=True);parser.add_argument('--samples',type=int,default=16)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
addon_utils.enable('photara_rs_engine')
from photara_rs_engine import engine
for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
assert bpy.ops.import_scene.photara_gaussians(filepath=str(Path(args.model).resolve()))=={'FINISHED'}
cloud=bpy.context.object
print('BLENDER_GAUSSIANS',len(cloud.data.points),flush=True)
config=json.loads(Path(args.camera).read_text());settings=config['camera']
bpy.ops.object.camera_add();camera=bpy.context.object
camera.matrix_world=Matrix(settings['matrix']);camera.data.lens=settings['lens']
for name in ('sensor_width','sensor_height','sensor_fit','clip_start','clip_end'):setattr(camera.data,name,settings[name])
scene=bpy.context.scene;scene.camera=camera
scene.render.engine='PHOTARA_RS';scene.photara.samples=args.samples
scene.photara.adaptive_sampling=False;scene.photara.denoise=False
scene.render.resolution_x,scene.render.resolution_y=config['resolution'];scene.render.resolution_percentage=100
scene.world.use_nodes=False;scene.world.color=(0,0,0)
scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
scene.view_settings.exposure=0;scene.view_settings.gamma=1
scene.render.image_settings.file_format='PNG';scene.render.filepath=str(Path(args.output).resolve())
frames=[];old=engine.PhotaraRenderEngine._fill_render_result
def capture(self,result,frame,layer):
 old(self,result,frame,layer);frames.append(frame.beauty.copy())
engine.PhotaraRenderEngine._fill_render_result=capture
bpy.ops.render.render(write_still=True)
assert len(frames)==1 and np.isfinite(frames[0]).all() and frames[0][...,:3].std()>.01
print('BLENDER_TRAINED_GSPLAT_RENDER_PASS',flush=True)
