"""Benchmark viewport callbacks during camera and object edits.

Run Blender with --python scripts/benchmark_blender_viewport.py -- <output.json>.
The benchmark creates a disposable scene in the running Blender process.
"""
import json
import os
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
if os.environ.get('PHOTARA_BENCH_DENOISER'):
 scene.photara.viewport_denoise = os.environ['PHOTARA_BENCH_DENOISER'] != 'NONE'
 if scene.photara.viewport_denoise:
  scene.photara.viewport_denoiser = os.environ['PHOTARA_BENCH_DENOISER']
area=next(a for a in bpy.context.screen.areas if a.type=='VIEW_3D')
area.spaces.active.region_3d.view_perspective='PERSP'
area.spaces.active.shading.use_scene_world_render=False
bpy.context.preferences.view.show_splash=False
stats={'view_draw':[],'view_update':[],'uploads':0}
latencies=[]
first_frames=[]
frame_counts={'camera':0,'object':0,'settle':0}
observed_updates=0
errors=[]
engine.PhotaraRenderEngine._report_viewport_failure=lambda self,exc: errors.append(str(exc))
old_invalidate=engine.ViewportState.invalidate
old_restart=engine.ViewportState.restart_render
def invalidate(self,depsgraph=None,**kwargs):
 if not self.needs_update:self._benchmark_invalidated=time.perf_counter()
 return old_invalidate(self,depsgraph,**kwargs)
def restart(self):
 requested=getattr(self,'_benchmark_invalidated',None)
 if requested is not None:
  latencies.append((time.perf_counter()-requested)*1000)
  self._benchmark_invalidated=None
 result=old_restart(self)
 self._benchmark_frame_started=time.perf_counter()
 if self.gpu_display:
  import _cycles
  self._benchmark_start_updates=_cycles.photara_display_stats(self._display_capsule)['gpu_updates']
 return result
engine.ViewportState.invalidate=invalidate
engine.ViewportState.restart_render=restart
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
    global observed_updates
    delta=native['gpu_updates']-observed_updates
    if delta>0:
     phase='camera' if ticks<30 else 'object' if ticks<60 else 'settle'
     frame_counts[phase]+=delta
     observed_updates=native['gpu_updates']
    started=getattr(viewport,'_benchmark_frame_started',None)
    if started is not None and native['gpu_updates']>viewport._benchmark_start_updates:
     first_frames.append((time.perf_counter()-started)*1000)
     viewport._benchmark_frame_started=None

   return result
  finally:stats[name].append((time.perf_counter()-t)*1000)
 return timed
for name in ('view_draw','view_update'):
 setattr(engine.PhotaraRenderEngine,name,make_timed(getattr(engine.PhotaraRenderEngine,name),name))
old_upload=engine.ViewportState._upload_result
def upload(self,frame):
 old_upload(self,frame);stats['uploads']+=1
engine.ViewportState._upload_result=upload
start=time.monotonic(); ticks=0;last=time.monotonic();gaps=[];settle_start=None
def tick():
 global ticks,last,settle_start
 if errors:
  print('Viewport benchmark failed: '+ '; '.join(errors),file=sys.stderr,flush=True)
  os._exit(1)
 now=time.monotonic();gaps.append((now-last)*1000);last=now
 if stats['uploads'] == 0 and not stats.get('gpu_updates',0) and now-start < 30:return .03
 if ticks<60:
  if ticks<30:
   r=area.spaces.active.region_3d;r.view_rotation=Quaternion((0,0,1),.02)@r.view_rotation
  else:
   cube.location.x += .01;bpy.context.view_layer.update()
  area.tag_redraw();ticks+=1;return .03
 if settle_start is None:settle_start=now
 if now-settle_start<1.0:return .03
 if now-start<60 and stats['uploads']<2 and stats.get('gpu_updates',0)<2:return .05
 result={'uploads':stats['uploads'],'gpu_updates':stats.get('gpu_updates',0),'cpu_maps':stats.get('cpu_maps',0),'seconds':now-start,'callback_ms':{}}
 for name in ('view_draw','view_update'):
  v=sorted(stats[name]);result['callback_ms'][name]={'max':max(v,default=0),'p95':v[int((len(v)-1)*.95)] if v else 0,'count':len(v)}
 result['timer_gap_max_ms']=max(gaps)
 result['frames_by_phase']=frame_counts
 result['denoiser']=scene.photara.viewport_denoiser if scene.photara.viewport_denoise else 'NONE'
 for label,values in [('invalidate_to_restart_ms',latencies),('restart_to_first_frame_ms',first_frames)]:
  values=sorted(values)
  result[label]={'count':len(values),'median':values[len(values)//2] if values else 0,'p95':values[int((len(values)-1)*.95)] if values else 0}

 args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
 path=Path(args[0]) if args else Path(__file__).resolve().parent.parent/'test_outputs/viewport_benchmark.json'
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(result,indent=2));print('BENCHMARK',json.dumps(result),flush=True)
 bpy.ops.wm.quit_blender()
bpy.app.timers.register(tick,first_interval=.1)
area.spaces.active.shading.type='RENDERED'
