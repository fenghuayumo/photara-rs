"""Verify sampling controls against the installed Blender Python package."""
import math
import photara_rs as pr
with pr.Scene(64,64,samples=1048576,denoise=False,adaptive_sampling=False) as scene:
 scene.configure_sampling(noise_threshold=.02,min_samples=16,time_limit=.1,denoise_start_sample=8)
 for values in ({'noise_threshold':float('nan')},{'min_samples':-1},{'time_limit':-1},{'denoise_start_sample':-1}):
  try:scene.configure_sampling(**values)
  except RuntimeError:pass
  else:raise AssertionError(values)
 job=scene.render_async()
 frame=job.wait()
 progress=job.poll()
 assert progress.state=='done',progress
 assert progress.current_sample<1048576,progress
 print('SAMPLING_CONTROLS_PASS',progress.current_sample,flush=True)
