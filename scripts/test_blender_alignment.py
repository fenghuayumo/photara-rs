# Dispatch to the actual display backend's pixel-level alignment check.
from pathlib import Path
import runpy
import _cycles

if hasattr(_cycles, 'photara_display_create'):
    runpy.run_path(str(Path(__file__).with_name('test_blender_gpu_display.py')), run_name='__main__')
else:
    """Compare rendered cube bounds with Blender's viewport projection at three poses.

    Run Blender with --python scripts/test_blender_alignment.py --python-exit-code 1.
    Uses a disposable scene and exits after validation.
    """
    import sys,time,json,traceback
    from pathlib import Path
    import bpy,numpy as np
    from mathutils import Vector,Quaternion
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'scripts'))
    from test_blender import prepare_scene
    from photara_rs_engine import engine
    scene,cube,material=prepare_scene()
    bpy.context.preferences.view.show_splash=False
    area=next(a for a in bpy.context.screen.areas if a.type=='VIEW_3D')
    area.spaces.active.region_3d.view_perspective='PERSP'
    reports=[];failures=[]
    old=engine.ViewportState._upload_result
    def upload(self,frame):
     old(self,frame)
     if self.pending_samples != scene.photara_rs.viewport_samples:return
     ctx=bpy.context;rd=ctx.region_data;region=ctx.region
     xy=[location_3d_to_region_2d(region,rd,cube.matrix_world@Vector(c)) for c in cube.bound_box]
     expected=[min(p.x for p in xy),min(p.y for p in xy),max(p.x for p in xy),max(p.y for p in xy)]
     ids=np.flipud(self.scene._session.object_id());y,x=np.where(ids==7)
     actual=[int(x.min()),int(y.min()),int(x.max()),int(y.max())] if len(x) else []
     error=max(abs(a-b) for a,b in zip(expected,actual)) if actual else 999
     reports.append(dict(expected=expected,actual=actual,error=error))
     print('ALIGN',reports[-1],flush=True)
    engine.ViewportState._upload_result=upload
    engine.PhotaraRenderEngine._report_viewport_failure=lambda self,exc:failures.append(str(exc))
    start=time.monotonic();stage=0
    def tick():
     global stage
     try:
      assert not failures,failures
      if time.monotonic()-start>30:raise AssertionError('timeout')
      if len(reports)>stage:
       if stage==2:
        (Path(__file__).resolve().parent.parent/'test_outputs/alignment.json').write_text(json.dumps(reports,indent=2))
        assert max(r['error'] for r in reports)<4,reports
        print('ALIGNMENT_PASS',flush=True);bpy.ops.wm.quit_blender();return None
       stage+=1
       r=area.spaces.active.region_3d;r.view_rotation=Quaternion((0,0,1),.5)@r.view_rotation;r.view_location+=Vector((.4,.3,.2));area.tag_redraw()
      return .05
     except Exception:
      traceback.print_exc();sys.stdout.flush();import os;os._exit(1)
    bpy.app.timers.register(tick,first_interval=.1)
    area.spaces.active.shading.type='RENDERED'
