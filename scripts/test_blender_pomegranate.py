"""Load assets_room.blend with Blender, then run this script to verify packed peel color."""
import sys
from pathlib import Path
import bpy,numpy as np
root=Path(__file__).resolve().parent.parent
if '--installed' in sys.argv:
 from photara_rs_engine import sync
 import photara_rs as pr
 assert root not in Path(sync.__file__).resolve().parents, sync.__file__
 assert hasattr(sync, '_image_filepath'), sync.__file__
else:
 sys.path[:0]=[str(root),str(root/'photara/python')]
 import photara_rs as pr
 import importlib.util
 spec=importlib.util.spec_from_file_location('verified_sync',root/'photara_rs_engine/sync.py')
 sync=importlib.util.module_from_spec(spec)
 spec.loader.exec_module(sync)
print('SYNC_SOURCE',sync.__file__,flush=True)
obj=bpy.data.objects['Pomegranate_Peel']
with pr.Scene(256,256,samples=32,denoise=True,adaptive_sampling=False) as scene:
 state=sync.SceneSyncState(scene)
 mesh=state._sync_mesh_data(obj.data,obj)
 scene.objects.new('Peel',data=mesh,object_id=7)
 v=np.array([x.co[:] for x in obj.data.vertices])
 center=(v.min(axis=0)+v.max(axis=0))/2
 radius=np.max(v.max(axis=0)-v.min(axis=0))
 scene.camera=pr.Camera(position=center+np.array([0,-radius*2.5,radius*.5]),target=center,up=(0,0,1),fov=35)
 scene.world.color=(0.7,0.7,0.7)
 scene.world.strength=1
 frame=scene.render()
 pixels=frame.beauty[frame.object_id==7,:3]
 mean=pixels.mean(axis=0)
 print('PEEL_MEAN_RGB',mean,flush=True)
 frame.save_png(str(root/'test_outputs/pomegranate_packed_verified.png'))
 assert mean[0]>mean[1]*1.3,mean
 print('POMEGRANATE_PACKED_PASS',flush=True)
