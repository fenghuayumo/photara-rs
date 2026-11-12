"""Render a trained 3DGS cloud using a saved Blender reference camera."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import photara_rs as pr
from photara_rs.gsplat import load_gaussians
parser=argparse.ArgumentParser();parser.add_argument('ply');parser.add_argument('--camera',required=True);parser.add_argument('--output',required=True);parser.add_argument('--samples',type=int,default=16)
args=parser.parse_args();start=time.perf_counter()
config=json.loads(Path(args.camera).read_text());camera=config['camera'];width,height=config['resolution'];aspect=width/height
matrix=np.asarray(camera['matrix'],np.float32)@np.diag([1,1,-1,1]).astype(np.float32)
fov=math.degrees(2*math.atan(camera['sensor_width']/(2*camera['lens']*aspect)))
print('Loading cloud',flush=True);data=load_gaussians(args.ply)
n=len(data['positions']);print('Loaded',n,'Gaussians in',time.perf_counter()-start,'seconds',flush=True)
with pr.Scene(width,height,samples=args.samples,denoise=False,adaptive_sampling=False) as scene:
 scene.world.color=(0,0,0)
 scene.camera=pr.Camera(matrix=matrix,fov=fov,near_clip=camera['clip_start'],far_clip=camera['clip_end'])
 print('Uploading',flush=True);scene.add_gaussian_splats(**data,object_id=1);del data
 print('Rendering',flush=True);frame=scene.render()
 assert np.isfinite(frame.beauty).all()
 assert frame.beauty[...,:3].std()>.01,'Blank image'
 frame.save_png(args.output)
 elapsed=time.perf_counter()-start
 Path(args.output).with_suffix('.json').write_text(json.dumps(dict(source=args.ply,count=n,resolution=[width,height],samples=args.samples,seconds=elapsed),indent=2))
 print('TRAINED_GSPLAT_RENDER_PASS',args.output,elapsed,flush=True)
