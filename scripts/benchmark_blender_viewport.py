"""Benchmark viewport callbacks during camera and object edits.

Run Blender with --python scripts/benchmark_blender_viewport.py -- <output.json>.
The benchmark creates a disposable scene in the running Blender process.
"""
import json
import time
from pathlib import Path
import sys
import bpy
from mathutils import Quaternion
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from test_blender import prepare_scene
from photara_rs_engine import engine
scene,cube,material=prepare_scene()
scene.photara.viewport_samples=64
scene.photara.viewport_denoise=True
area=next(a for a in bpy.context.screen.areas if a.type=='VIEW_3D')
area.spaces.active.region_3d.view_perspective='PERSP'
area.spaces.active.shading.use_scene_world_render=False
bpy.context.preferences.view.show_splash=False
stats={'view_draw':[],'view_update':[],'uploads':0}
def make_timed(old,name):
 def timed(self,context,depsgraph):
  t=time.perf_counter()
  try:
   result=old(self,context,depsgraph)
   viewport=getattr(self,'_viewport',None)
   if name=='view_draw' and viewport and viewport.gpu_display:
    import _cycles
    native=_cycles.photara_display_stats(viewport._display_capsule)
    stats.update(native)
   return result
  finally:stats[name].append((time.perf_counter()-t)*1000)
 return timed
for name in ('view_draw','view_update'):
 setattr(engine.PhotaraRenderEngine,name,make_timed(getattr(engine.PhotaraRenderEngine,name),name))
old_upload=engine.ViewportState._upload_result
def upload(self,frame):
 old_upload(self,frame);stats['uploads']+=1
engine.ViewportState._upload_result=upload
start=time.monotonic(); ticks=0;last=time.monotonic();gaps=[]
def tick():
 global ticks,last
 now=time.monotonic();gaps.append((now-last)*1000);last=now
 if stats['uploads'] == 0 and not stats.get('gpu_updates',0) and now-start < 30:return .03
 if ticks<60:
  if ticks<30:
   r=area.spaces.active.region_3d;r.view_rotation=Quaternion((0,0,1),.02)@r.view_rotation
  else:
   cube.location.x += .01;bpy.context.view_layer.update()
  area.tag_redraw();ticks+=1;return .03
 if now-start<60 and stats['uploads']<2 and stats.get('gpu_updates',0)<2:return .05
 result={'uploads':stats['uploads'],'gpu_updates':stats.get('gpu_updates',0),'cpu_maps':stats.get('cpu_maps',0),'seconds':now-start,'callback_ms':{}}
 for name in ('view_draw','view_update'):
  v=sorted(stats[name]);result['callback_ms'][name]={'max':max(v,default=0),'p95':v[int((len(v)-1)*.95)] if v else 0,'count':len(v)}
 result['timer_gap_max_ms']=max(gaps)
 args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
 path=Path(args[0]) if args else Path(__file__).resolve().parent.parent/'test_outputs/viewport_benchmark.json'
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(result,indent=2));print('BENCHMARK',json.dumps(result),flush=True)
 bpy.ops.wm.quit_blender()
bpy.app.timers.register(tick,first_interval=.1)
area.spaces.active.shading.type='RENDERED'
