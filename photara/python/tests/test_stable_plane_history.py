"""GPU plane history through a Principled glass slab using the public API."""
import unittest
import os
from unittest.mock import patch
import photara_rs as pr
import numpy as np
from test_history_samples import plane_scene

class StablePlaneHistoryTests(unittest.TestCase):
    def test_automatic_build_emission_and_no_double_count(self):
        # Observe deterministic radiance before NGX, independently of explicit
        # history_samples. White glass has analytic normal-incidence T=0.96.
        passes = 'combined depth normal albedo object_id stable_radiance'
        with patch.dict(os.environ, PHOTARA_PASSES=passes):
            for glass_enabled, glass_roughness in ((False,0), (True,0), (True,0.001)):
                with pr.Scene(64, 48, samples=1, adaptive_sampling=False,
                              denoise=True, denoiser='DLSS') as scene:
                    scene.camera.location=(0,0,-4)
                    scene.camera.target=(0,0,0)
                    scene.world.color=(0,0,0)
                    emission=scene.materials.new('emitter', base_color=(0,0,0), roughness=1,
                        emission_color=(0.2,0.4,0.6), emission_strength=2)
                    def quad(name,z,material,reverse=False):
                        points=np.array([(-10,-10,z),(-10,10,z),(10,10,z),(10,-10,z)],np.float32)
                        faces=np.array([(0,2,1),(0,3,2)] if reverse else [(0,1,2),(0,2,3)],np.int32)
                        mesh=scene.meshes.new(name,points,faces,material=material)
                        scene.objects.new(name,mesh)
                    quad('emitter',0,emission)
                    if glass_enabled:
                        glass=scene.materials.new('glass',base_color=(1,1,1),roughness=glass_roughness,
                                                   transmission=1,ior=1.5)
                        quad('front',-1,glass)
                        quad('back',-0.9,glass,True)
                    images=[]
                    for seed in (1,19):
                        image=scene.render(seed=seed).rgb[20:28,28:36]
                        stable=scene._session.get_pass('stable_radiance')[20:28,28:36]
                        self.assertTrue(np.isfinite(stable).all())
                        expected=np.array((0.4,0.8,1.2))*(0.96**2 if glass_enabled else 1)
                        np.testing.assert_allclose(stable.mean(axis=(0,1)),expected,rtol=0.003,atol=0.001)
                        # A duplicate stochastic emission write would nearly double
                        # this constant signal. Leave margin for NGX reconstruction.
                        np.testing.assert_allclose(image.mean(axis=(0,1)),expected,rtol=0.15,atol=0.025)
                        images.append(stable)
                    np.testing.assert_allclose(images[0],images[1],rtol=0.003,atol=0.001)
                    self.assertEqual(scene.history_samples,0)
                    scene.samples=2
                    scene.render()
                    self.assertEqual(float(scene._session.get_pass('stable_radiance').max()),0)

    def test_alpha_cutout_does_not_hide_uncaptured_emission(self):
        import struct
        import zlib
        import tempfile
        from pathlib import Path
        def chunk(name, data):
            return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, PHOTARA_DISABLE_OMM='1'):
            texture=Path(directory)/'transparent.png'
            texture.write_bytes(b'\x89PNG\r\n\x1a\n'+
                chunk(b'IHDR',struct.pack('>IIBBBBB',2,2,8,6,0,0,0))+
                chunk(b'IDAT',zlib.compress((b'\x00'+b'\xff\xff\xff\x00'*2)*2))+chunk(b'IEND',b''))
            with pr.Scene(64,48,samples=1,adaptive_sampling=False,denoise=True,denoiser='DLSS') as scene:
                scene.camera.location=(0,0,-4)
                scene.camera.target=(0,0,0)
                scene.world.color=(0,0,0)
                emitter=scene.materials.new('emitter',base_color=(0,0,0),roughness=1,
                    emission_color=(0.2,0.4,0.6),emission_strength=2)
                leaf=scene.materials.new('cutout',roughness=1)
                leaf.base_color_texture={'path':str(texture),'colorspace':'data'}
                leaf.configure_alpha_clip(0.5)
                for name,z,material in [('emitter',0,emitter),('cutout',-1,leaf)]:
                    mesh=scene.meshes.new(name,[(-10,-10,z),(-10,10,z),(10,10,z),(10,-10,z)],
                        [(0,1,2),(0,2,3)],uvs=[(0,0),(0,1),(1,1),(1,0)],material=material)
                    scene.objects.new(name,mesh)
                image=scene.render().rgb[20:28,28:36].mean(axis=(0,1))
                np.testing.assert_allclose(image,(0.4,0.8,1.2),rtol=0.15,atol=0.025)

    def test_glass_gpu_history_and_disable(self):
        with plane_scene(history_samples=0, denoise=True, denoiser='DLSS') as scene:
            scene.samples=1
            glass = scene.materials.new('glass', base_color=(1,1,1), roughness=0,
                                        transmission=1, ior=1.5)
            for i,z in enumerate((-1.0,-0.9)):
                points = np.array([(-10,-10,z),(-10,10,z),(10,10,z),(10,-10,z)],np.float32)
                triangles = np.array([(0,1,2),(0,2,3)] if i==0 else [(0,2,1),(0,3,2)],np.int32)
                mesh = scene.meshes.new('glass'+str(i),points,triangles,material=glass)
                scene.objects.new('glass'+str(i),mesh,object_id=10+i)
            scene.render()
            scene.history_samples=2
            scene.render()
            scene.camera.location=(0.01,0,-4)
            image=scene.render().beauty
            self.assertTrue(np.isfinite(image).all())
            self.assertGreater(scene.history_stats()['reused_pixels'],0)
            scene.reset_history()
            scene.render()
            self.assertEqual(scene.history_stats()['reused_pixels'],0)
            scene.history_samples=0
            self.assertTrue(np.isfinite(scene.render(seed=17).beauty).all())

if __name__=='__main__': unittest.main()
