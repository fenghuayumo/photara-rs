"""Exercise DLSS native resolution and every supported upscale mode on a GPU."""
import unittest
import numpy as np
import photara_rs as pr


class DLSSTests(unittest.TestCase):
    def test_all_quality_modes(self):
        for mode in ('NONE', 'QUALITY', 'BALANCED', 'PERFORMANCE', 'ULTRA_PERFORMANCE'):
            with self.subTest(mode=mode):
                try:
                    scene = pr.Scene(160, 120, samples=4, device='OPTIX',
                                     denoiser='DLSS', denoise_upscale_quality=mode)
                except Exception as exc:
                    if 'unavailable' in str(exc):
                        self.skipTest(str(exc))
                    raise
                with scene:
                    scene.world.color = (0.2, 0.1, 0.05)
                    image = scene.render().beauty
                    self.assertEqual(image.shape, (120, 160, 4))
                    self.assertTrue(np.isfinite(image).all())
                    self.assertGreater(float(image[..., :3].max()), 0.01)

    def test_invalid_quality(self):
        with self.assertRaises(ValueError):
            pr.Scene(16, 16, denoise_upscale_quality='INVALID')
