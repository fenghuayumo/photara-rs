"""Exercise the public denoiser configuration with actual geometry."""

import unittest

import numpy as np
import photara_rs as pr


class DenoiserTests(unittest.TestCase):
    def test_explicit_denoisers(self):
        for kind, use_gpu in [('OPTIX', True), ('OPENIMAGEDENOISE', False),
                              ('OPENIMAGEDENOISE', True)]:
            with self.subTest(denoiser=kind, use_gpu=use_gpu):
                with pr.Scene(64, 48, samples=4, denoise=True,
                              adaptive_sampling=False, denoiser=kind,
                              denoise_use_gpu=use_gpu) as scene:
                    material = scene.materials.new('red', base_color=(0.8, 0.1, 0.05))
                    mesh = scene.meshes.new('triangle', [(-2, -2, 0), (2, -2, 0), (0, 2, 0)],
                                            [(0, 1, 2)], material=material)
                    scene.objects.new('triangle', mesh, object_id=7)
                    scene.camera.location = (0, 0, -4)
                    scene.camera.target = (0, 0, 0)
                    scene.world.color = (0.5, 0.5, 0.5)
                    frame = scene.render()
                    self.assertTrue(np.isfinite(frame.beauty).all())
                    self.assertTrue((frame.object_id == 7).any())

    def test_invalid_configuration(self):
        for setting in ('denoiser', 'denoise_passes', 'denoise_prefilter', 'denoise_quality'):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                pr.Scene(**{setting: 'INVALID'})
