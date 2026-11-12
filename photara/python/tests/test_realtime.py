"""RealTime frame lifecycle and Reference isolation on the actual OptiX backend."""
import unittest

import numpy as np

import photara_rs as pr
from test_async_render import build_scene


class RealTimeTests(unittest.TestCase):
    def test_sequence_reset_and_reference_isolation(self):
        scene, _ = build_scene(samples=4)
        with scene:
            reference = scene.render(seed=23).beauty.copy()
            scene.configure_render_mode("REALTIME", samples_per_frame=1)
            job = scene.render_async()
            first = job.wait().beauty.copy()
            self.assertEqual(job.poll().num_samples, 1)
            second = scene.render().beauty.copy()
            self.assertGreater(float(np.max(np.abs(first - second))), 0)
            scene.configure_render_mode("REALTIME", samples_per_frame=1)
            np.testing.assert_array_equal(first, scene.render().beauty)
            scene.configure_render_mode("REFERENCE")
            job = scene.render_async()
            # Wavefront accumulation order can change at float roundoff scale.
            np.testing.assert_allclose(reference, job.wait().beauty, rtol=1e-6, atol=1e-7)
            self.assertEqual(job.poll().num_samples, 4)
            self.assertEqual(scene.samples, 4)

    def test_constructor_budget_and_resize(self):
        with pr.Scene(32, 24, samples=9, denoise=False, adaptive_sampling=False,
                      render_mode="REALTIME", samples_per_frame=2) as scene:
            job = scene.render_async()
            self.assertEqual(job.wait().beauty.shape, (24, 32, 4))
            self.assertEqual(job.poll().num_samples, 2)
            scene.resize(40, 30)
            self.assertEqual(scene.render().beauty.shape, (30, 40, 4))
            scene.configure_render_mode("REFERENCE")
            job = scene.render_async()
            job.wait()
            self.assertEqual(job.poll().num_samples, 9)

    def test_running_job_guard_and_cancel(self):
        scene, _ = build_scene(width=512, height=384, samples=4)
        with scene:
            scene.configure_render_mode("REALTIME", samples_per_frame=64)
            job = scene.render_async()
            with self.assertRaises(RuntimeError):
                scene.configure_render_mode("REFERENCE")
            job.cancel()
            scene.configure_render_mode("REFERENCE")
            job = scene.render_async()
            job.wait()
            self.assertEqual(job.poll().num_samples, 4)

    def test_validation_does_not_change_mode(self):
        with pr.Scene(16, 16, samples=3, denoise=False) as scene:
            for mode, budget in [("INVALID", 1), ("REALTIME", 0),
                                 ("REALTIME", 65), ("REALTIME", 1.5)]:
                with self.subTest(mode=mode, budget=budget), self.assertRaises(ValueError):
                    scene.configure_render_mode(mode, samples_per_frame=budget)
            # Exercise C validation as well as Python argument validation.
            for mode, budget in [(7, 1), (1, 0), (1, 65)]:
                with self.assertRaises(RuntimeError):
                    scene._session._ctx.set_render_mode(mode, budget)
            job = scene.render_async()
            job.wait()
            self.assertEqual(job.poll().num_samples, 3)


if __name__ == "__main__":
    unittest.main()
