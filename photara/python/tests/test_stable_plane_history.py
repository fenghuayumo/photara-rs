"""GPU plane history through a Principled glass slab using the public API."""
import unittest
import numpy as np
from test_history_samples import plane_scene

class StablePlaneHistoryTests(unittest.TestCase):
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
