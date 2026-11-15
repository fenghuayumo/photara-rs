"""macOS integration checks: real Metal renders, edits and custom GPU kernels."""
import sys
import platform
import unittest

import numpy as np


@unittest.skipUnless(sys.platform == 'darwin' and platform.machine() == 'arm64',
                     'requires Apple Silicon Metal')
class MetalTests(unittest.TestCase):
    def make_scene(self, **kwargs):
        import photara_rs as pr
        scene = pr.Scene(32, 24, samples=4, device=kwargs.pop('device', 'METAL'),
                         denoise=kwargs.pop('denoise', False), adaptive_sampling=False, **kwargs)
        self.addCleanup(scene.close)
        scene.camera.location = (0, 2, -3)
        scene.camera.target = (0, 0, 1)
        scene.world.color = (0.01, 0.01, 0.01)
        material = scene.materials.new('floor', base_color=(0.6, 0.3, 0.1))
        mesh = scene.meshes.new('floor', [(-3, 0, -1), (-3, 0, 5), (3, 0, 5), (3, 0, -1)],
                                [(0, 1, 2), (0, 2, 3)],
                                uvs=[(0, 0), (0, 1), (1, 1), (1, 0)], material=material)
        scene.objects.new('floor', mesh, object_id=7)
        scene.lights.new_area('key', position=(0, 2, 1), target=(0, 0, 1), strength=100)
        return scene, material

    def assert_frame(self, frame):
        self.assertEqual(frame.beauty.shape, (24, 32, 4))
        self.assertTrue(np.isfinite(frame.beauty).all())
        self.assertGreater(float(frame.rgb.mean()), 0)
        self.assertTrue(np.any(frame.object_id == 7))

    def test_render_edit_and_auto_selection(self):
        scene, material = self.make_scene(device='AUTO')
        self.assertIn('Apple', scene.device_name)
        first = scene.render()
        self.assert_frame(first)
        material.base_color = (0.1, 0.7, 0.2)
        second = scene.render()
        self.assert_frame(second)
        self.assertGreater(float(np.abs(first.rgb - second.rgb).mean()), 0.001)

    def test_denoising(self):
        import photara_rs as pr
        if pr.auto_denoiser() == 'NONE':
            self.skipTest('OpenImageDenoise disabled in this build')
        scene, _ = self.make_scene(denoise=True)
        self.assert_frame(scene.render())

    def test_neeat_and_temporal_gpu_kernels(self):
        scene, _ = self.make_scene(history_samples=2)
        for i in range(7):
            scene.lights.new_point(f'fill{i}', position=(5 + i, 5, 5), strength=0.1)
        scene.configure_render_mode('REALTIME', samples_per_frame=4)
        for _ in range(4):
            self.assert_frame(scene.render())
        stats = scene.neeat_stats()
        self.assertGreater(stats['gpu_updates'], 0)
        self.assertEqual(stats['cpu_updates'], 0)
        self.assertEqual(scene.history_stats()['backend'], 'GPU')
        self.assertGreater(scene.history_stats()['gpu_updates'], 0)

    def test_image_texture(self):
        import tempfile
        from pathlib import Path
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is required to create a test texture')
        scene, material = self.make_scene()
        with tempfile.TemporaryDirectory() as directory:
            texture = Path(directory) / 'green.png'
            Image.fromarray(np.full((32, 32, 3), (0, 255, 0), dtype=np.uint8)).save(texture)
            material.base_color_texture = {'path': str(texture)}
            frame = scene.render()
            self.assert_frame(frame)
            floor = frame.rgb[frame.object_id == 7]
            self.assertGreater(float(floor[:, 1].mean()), float(floor[:, 0].mean()) * 2)
