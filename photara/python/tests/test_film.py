"""Film controls must affect native pixels and alpha."""

import unittest

import numpy as np
import photara_rs as pr


class FilmTests(unittest.TestCase):
    def test_exposure_transparency_and_filters(self):
        with pr.Scene(32, 24, samples=2, denoise=False, adaptive_sampling=False) as scene:
            scene.world.color = (0.2, 0.3, 0.4)
            scene.configure_film(exposure=1, filter_type='BOX')
            first = scene.render().beauty
            scene.configure_film(exposure=2, filter_type='GAUSSIAN')
            second = scene.render().beauty
            np.testing.assert_allclose(second[..., :3], first[..., :3] * 2, atol=1e-5)
            self.assertTrue((first[..., 3] == 1).all())
            scene.configure_film(transparent=True, filter_type='BLACKMAN_HARRIS')
            transparent = scene.render().beauty
            self.assertTrue((transparent[..., 3] == 0).all())
            self.assertTrue(np.isfinite(transparent).all())
            with self.assertRaises(RuntimeError):
                scene.configure_film(exposure=-1)
