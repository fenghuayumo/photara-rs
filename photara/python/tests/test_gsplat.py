"""End-to-end 3DGS import, rendering and lifecycle checks on OptiX."""
from pathlib import Path
import tempfile, unittest
import numpy as np
import photara_rs as pr
from photara_rs.gsplat import load_gaussians, validate_gaussians, C0

class GSplatTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def fixture(self,fmt='ascii'):
        names=['x','y','z']+[f'f_dc_{i}' for i in range(3)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]+[f'f_rest_{i}' for i in range(45)]
        rows=np.zeros((2,len(names)),np.float32)
        rows[:,:3]=[[-.35,0,0],[.35,0,0]]
        rows[:,3:6]=(np.array([[1,.01,.01],[.01,.01,1]])-.5)/C0
        rows[:,6]=8
        rows[:,7:10]=np.log([.2,.25,.15])
        rows[:,10]=1
        rows[:,14:]=np.arange(45)*.0001
        path=self.root/(fmt+'.ply')
        with path.open('wb') as f:
            f.write(('ply\nformat '+fmt+' 1.0\nelement vertex 2\n'+''.join('property float '+n+'\n' for n in names)+'end_header\n').encode())
            if fmt=='ascii': np.savetxt(f,rows,fmt='%.9g')
            else:f.write(rows.astype('<f4' if fmt=='binary_little_endian' else '>f4').tobytes())
        return path
    def test_ply_encodings(self):
        reference=load_gaussians(self.fixture())
        for fmt in ('binary_little_endian','binary_big_endian'):
            data=load_gaussians(self.fixture(fmt))
            for key in data: np.testing.assert_allclose(data[key],reference[key],atol=1e-6)
        self.assertEqual(reference['sh_rest'].shape,(2,15,3))
        self.assertAlmostEqual(reference['sh_rest'][0,1,2],.0031,places=6)
        self.assertAlmostEqual(reference['scales'][0,0],.2,places=6)
    def test_splat_and_validation(self):
        dtype=np.dtype([('p','<f4',3),('s','<f4',3),('c','u1',4),('q','u1',4)])
        a=np.zeros(1,dtype);a['s']=.2;a['c']=[255,0,0,128];a['q']=[255,128,128,128]
        path=self.root/'one.splat';a.tofile(path)
        data=load_gaussians(path)
        self.assertAlmostEqual(data['opacities'][0],128/255)
        with self.assertRaises(ValueError):validate_gaussians([[0,0,0]],[[-1,1,1]])
        with self.assertRaises(ValueError):validate_gaussians([[0,0,0]],[[1,1,1]],rotations=[[0,0,0,0]])
        path.write_bytes(b'x')
        with self.assertRaises(ValueError):load_gaussians(path)
    def test_render_and_lifecycle(self):
        with pr.Scene(128,96,samples=8,denoise=False,adaptive_sampling=False) as scene:
            scene.world.color=(0,0,0)
            cloud=scene.load_gaussian_splats(self.fixture('binary_little_endian'))
            frame=scene.render();rgb=frame.beauty[...,:3]
            self.assertTrue(np.isfinite(rgb).all())
            self.assertGreater(((rgb[...,0]>.1)&(rgb[...,2]<.05)).sum(),20)
            self.assertGreater(((rgb[...,2]>.1)&(rgb[...,0]<.05)).sum(),20)
            cloud.set_visible(False)
            self.assertLess(scene.render().beauty[...,:3].max(),.001)
            cloud.set_visible(True)
            transform=np.eye(4,dtype=np.float32);transform[0,3]=.5
            cloud.set_transform(transform)
            self.assertFalse(np.allclose(frame.beauty,scene.render().beauty))
            cloud.remove()
            self.assertLess(scene.render().beauty[...,:3].max(),.001)
            with self.assertRaises(ValueError):cloud.set_visible(True)
    def test_anisotropic_rotation_and_opacity(self):
        images=[]
        for q in ((1,0,0,0),(2**-.5,0,0,2**-.5)):
            with pr.Scene(96,96,samples=8,denoise=False,adaptive_sampling=False) as scene:
                scene.world.color=(0,0,0)
                scene.add_gaussian_splats([[0,0,0]],[[.5,.08,.08]],rotations=[q],colors=[[1,0,0]],opacities=[.5])
                image=scene.render().beauty
                images.append(image)
                self.assertGreater(image[48,48,0],.15)
                self.assertLess(image[48,48,0],.3)
        y,x=np.where(images[0][...,0]>.02)
        self.assertGreater(x.ptp() if hasattr(x,'ptp') else np.ptp(x),np.ptp(y)*2)
        y,x=np.where(images[1][...,0]>.02)
        self.assertGreater(np.ptp(y),np.ptp(x)*2)

    def test_shared_instances_survive_source_removal(self):
        with pr.Scene(64,64,samples=2,denoise=False,adaptive_sampling=False) as scene:
            scene.world.color=(0,0,0)
            cloud=scene.add_gaussian_splats([[0,0,0]],[[.15,.15,.15]],colors=[[1,0,0]])
            handles=[]
            for i in range(16):
                matrix=np.eye(4,dtype=np.float32)
                matrix[0,3]=(i-8)*.08
                handles.append(scene._session.add_instance(cloud.handle,transform=matrix))
            cloud.remove()
            self.assertGreater(scene.render().beauty[...,0].max(),.1)
            for handle in handles:scene._session.remove_object(handle)
            self.assertLess(scene.render().beauty[...,:3].max(),.001)

    def test_mesh_and_dlss(self):
        for denoiser in ('OPTIX','OPENIMAGEDENOISE','DLSS'):
            with pr.Scene(64,64,samples=4,denoiser=denoiser,adaptive_sampling=False,denoise_upscale_quality='QUALITY' if denoiser=='DLSS' else 'NONE') as scene:
                scene.world.color=(.1,.1,.1)
                scene.load_gaussian_splats(self.fixture())
                mesh=scene.meshes.new('backdrop',[(-3,-3,1),(3,-3,1),(0,3,1)],[(0,1,2)])
                scene.objects.new('backdrop',mesh)
                frame=scene.render()
                self.assertTrue(np.isfinite(frame.beauty).all())
                self.assertGreater(frame.beauty[...,0].max(),.1)
if __name__=='__main__':unittest.main()
