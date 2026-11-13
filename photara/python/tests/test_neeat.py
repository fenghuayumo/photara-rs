"""Feedback learning, mixture support and mode isolation on the OptiX device."""
import unittest

import numpy as np
import photara_rs as pr


def scene_fixture():
    scene = pr.Scene(32, 24, samples=1024, max_bounces=3,
                     denoise=False, adaptive_sampling=False)
    scene.camera.location = (0, 2, -3)
    scene.camera.target = (0, 0, 1)
    scene.world.color = (0, 0, 0)
    matte = scene.materials.new("matte", base_color=(.6, .6, .6), roughness=.9)
    mesh = scene.meshes.new("floor", [(-3, 0, -1), (-3, 0, 5), (3, 0, 5), (3, 0, -1)],
                            [(0, 1, 2), (0, 2, 3)], material=matte)
    scene.objects.new("floor", mesh)
    scene.lights.new_area("key", position=(0, 2, 1), target=(0, 0, 1),
                          size=(1, 1), strength=80)
    for i in range(7):
        scene.lights.new_area(f"far{i}", position=(10+i, 10, 10), target=(0, 0, 1),
                              size=(1, 1), strength=.1)
    return scene


class NEEATTests(unittest.TestCase):
    def test_camera_motion_preserves_learned_proposal(self):
        with scene_fixture() as scene:
            scene.configure_render_mode("REALTIME", samples_per_frame=8)
            for _ in range(24):
                scene.render()
            self.assertGreater(scene.neeat_stats()["max_pdf"], .6)
            for i in range(8):
                scene.camera.location = (.01 * (i + 1), 2, -3)
                self.assertTrue(np.isfinite(scene.render().rgb).all())
                self.assertGreater(scene.neeat_stats()["max_pdf"], .6)
            # Explicit reconfiguration still invalidates the proposal.
            scene.configure_render_mode("REALTIME", samples_per_frame=8)
            scene.render()
            self.assertAlmostEqual(scene.neeat_stats()["max_pdf"], 1 / 8, places=5)

    def test_mesh_dominant_scene_keeps_tree_budget(self):
        with pr.Scene(64, 64, samples=4096, max_bounces=2,
                      denoise=False, adaptive_sampling=False) as scene:
            scene.camera.location = (0, 2.4, -4)
            scene.camera.target = (0, 0, 1)
            scene.world.color = (0, 0, 0)
            matte = scene.materials.new("floor", base_color=(.7, .7, .7), roughness=.8)
            glow = scene.materials.new("glow", base_color=(0, 0, 0),
                                       emission_color=(1, 1, 1), emission_strength=80)
            for name, vertices, material in [
                ("floor", [(-3, 0, -1), (-3, 0, 5), (3, 0, 5), (3, 0, -1)], matte),
                ("emitter", [(-.25, 4, .75), (.25, 4, .75),
                             (.25, 4, 1.25), (-.25, 4, 1.25)], glow),
            ]:
                mesh = scene.meshes.new(name, vertices, [(0, 1, 2), (0, 2, 3)],
                                        material=material)
                scene.objects.new(name, mesh)
            weak = scene.lights.new_point("weak", position=(10, 10, 10), strength=.0001)
            reference = scene.render().rgb.copy()
            errors, cold_frames = [], []
            for enabled in (False, True):
                scene.configure_neeat(enabled)
                scene.configure_render_mode("REALTIME", samples_per_frame=1)
                cold_frames.append(scene.render_async().wait(poll_interval=.001).rgb.copy())
                for _ in range(31):
                    scene.render_async().wait(poll_interval=.001)
                frames = np.asarray([scene.render_async().wait(poll_interval=.001).rgb.copy()
                                     for _ in range(128)])
                self.assertTrue(np.isfinite(frames).all())
                self.assertLess(abs(float(frames.mean() / reference.mean()) - 1), .03)
                errors.append(float(np.mean((frames - reference) ** 2)))
            self.assertLess(errors[1], 1.3 * errors[0])
            np.testing.assert_allclose(cold_frames[0], cold_frames[1], rtol=1e-5, atol=1e-6)
            self.assertEqual(scene.neeat_stats()["active_lights"], 1)
            # A real lighting change must still reset and learn an analytic
            # proposal, rather than permanently disabling NEE-AT for all meshes.
            weak.location = (0, 2, 1)
            weak.strength = 100
            for _ in range(16):
                self.assertTrue(np.isfinite(scene.render().rgb).all())

    def test_learning_energy_reset_and_reference_isolation(self):
        with scene_fixture() as scene:
            reference = scene.render().rgb.copy()
            self.assertEqual(scene.neeat_stats()["active_lights"], 0)
            scene.configure_render_mode("REALTIME", samples_per_frame=64)
            first = scene.render().rgb.copy()
            self.assertEqual(scene.neeat_stats()["active_lights"], 8)
            self.assertAlmostEqual(scene.neeat_stats()["max_pdf"], 1/8, places=5)
            frames = [scene.render().rgb.copy() for _ in range(24)]
            self.assertGreater(scene.neeat_stats()["max_pdf"], .6)
            mean = np.mean(frames, axis=0)
            relative_rmse = np.sqrt(np.mean((mean-reference)**2)) / reference.mean()
            self.assertLess(relative_rmse, .12)
            # A reset discards feedback and reproduces the initial proposal.
            scene.configure_render_mode("REALTIME", samples_per_frame=64)
            np.testing.assert_allclose(first, scene.render().rgb, rtol=1e-5, atol=1e-6)
            scene.configure_neeat(False)
            scene.render()
            self.assertEqual(scene.neeat_stats()["active_lights"], 0)
            scene.configure_neeat(True)
            scene.configure_render_mode("REFERENCE")
            np.testing.assert_allclose(reference, scene.render().rgb, rtol=1e-5, atol=1e-6)
            self.assertEqual(scene.neeat_stats()["active_lights"], 0)

    def test_screen_tiles_learn_distinct_lights_and_resize(self):
        with pr.Scene(96, 48, samples=2048, max_bounces=2,
                      denoise=False, adaptive_sampling=False) as scene:
            scene.camera.location = (0, 5, -5)
            scene.camera.target = (0, 0, 0)
            scene.camera.fov = 60
            scene.world.color = (0, 0, 0)
            matte = scene.materials.new("floor", base_color=(.7, .7, .7), roughness=1)
            mesh = scene.meshes.new("floor", [(-6, 0, -4), (-6, 0, 6),
                                               (6, 0, 6), (6, 0, -4)],
                                    [(0, 1, 2), (0, 2, 3)], material=matte)
            scene.objects.new("floor", mesh)
            for x in (-2.5, 2.5):
                scene.lights.new_point(str(x), position=(x, .7, 0), strength=100)
            reference = scene.render().rgb.copy()
            scene.configure_render_mode("REALTIME", samples_per_frame=64)
            scene.render()
            self.assertEqual(scene.neeat_stats()["local_tiles"], 6)
            self.assertAlmostEqual(scene.neeat_stats()["local_max_pdf"], .5, places=5)
            frames = [scene.render().rgb.copy() for _ in range(32)]
            stats = scene.neeat_stats()
            self.assertGreater(stats["local_max_pdf"], stats["max_pdf"] + .15)
            mean = np.mean(frames[-16:], axis=0)
            self.assertLess(np.sqrt(np.mean((mean-reference)**2)) / reference.mean(), .12)
            # Same emitter count, different screen layout: discard local history.
            scene.resize(48, 96)
            scene.render()
            self.assertEqual(scene.neeat_stats()["local_tiles"], 6)
            self.assertAlmostEqual(scene.neeat_stats()["local_max_pdf"], .5, places=5)
            scene.configure_neeat(False)
            scene.render()
            self.assertEqual(scene.neeat_stats()["local_tiles"], 0)

    def test_emissive_mesh_fallback_and_reenable(self):
        with scene_fixture() as scene:
            scene.configure_render_mode("REALTIME", samples_per_frame=4)
            scene.render()
            scene.lights.clear()
            scene.materials[0].emission_color = (1, .5, .2)
            scene.materials[0].emission_strength = 2
            image = scene.render().rgb
            self.assertTrue(np.isfinite(image).all())
            self.assertGreater(float(image.mean()), 0)
            self.assertEqual(scene.neeat_stats()["active_lights"], 0)
            scene.lights.new_point("added", position=(0, 2, 1), strength=80)
            scene.render()
            self.assertEqual(scene.neeat_stats()["active_lights"], 1)

    def test_dynamic_light_rebuild_and_idle_guard(self):
        with scene_fixture() as scene:
            scene.configure_render_mode("REALTIME", samples_per_frame=4)
            for _ in range(8):
                scene.render()
            scene.lights[0].strength = .01
            scene.lights[1].strength = 40000
            scene.render()  # Rebuilt light table must start from a fresh proposal.
            self.assertAlmostEqual(scene.neeat_stats()["max_pdf"], 1/8, places=5)
            for _ in range(16):
                image = scene.render().rgb
                self.assertTrue(np.isfinite(image).all())
            self.assertGreater(scene.neeat_stats()["max_pdf"], .5)
            job = scene.render_async()
            with self.assertRaises(RuntimeError):
                scene.configure_neeat(False)
            job.wait()


if __name__ == "__main__":
    unittest.main()
