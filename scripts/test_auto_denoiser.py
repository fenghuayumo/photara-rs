"""Check installed Automatic denoiser selection and explicit OIDN agreement."""
import photara_rs as pr
import numpy as np
assert pr.auto_denoiser()=='OPENIMAGEDENOISE',pr.auto_denoiser()
images=[]
for denoiser in ('AUTO','OPENIMAGEDENOISE'):
 with pr.Scene(48,48,samples=8,seed=7,denoiser=denoiser,adaptive_sampling=False) as scene:
  material=scene.materials.new('red',base_color=(.8,.05,.02))
  mesh=scene.meshes.new('triangle',[(-1,-1,0),(1,-1,0),(0,1,0)],[(0,1,2)],material=material)
  scene.objects.new('triangle',mesh)
  scene.world.color=(.7,.7,.7)
  images.append(scene.render().beauty.copy())
assert np.isfinite(images).all()
assert np.allclose(images[0],images[1],rtol=1e-4,atol=1e-5)
print('AUTO_OIDN_RENDER_PASS')
