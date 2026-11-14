"""Actual GPU renders exercising the denoiser-independent bounded raw-frame history."""
import unittest
import os
from unittest.mock import patch

import numpy as np
import photara_rs as pr


def plane_scene(**kwargs):
    scene = pr.Scene(64, 48, samples=4, adaptive_sampling=kwargs.pop("adaptive_sampling", False),
                     denoise=kwargs.pop("denoise", False), **kwargs)
    mat = scene.materials.new("wall", base_color=(0.5, 0.3, 0.1), roughness=1.0)
    mesh = scene.meshes.new("wall", np.array([
        (-10, -10, 0), (-10, 10, 0), (10, 10, 0), (10, -10, 0)], np.float32),
        np.array([(0, 1, 2), (0, 2, 3)], np.int32), material=mat)
    scene.objects.new("wall", mesh, object_id=1)
    scene.camera.location = (0, 0, -4)
    scene.camera.target = (0, 0, 0)
    scene.lights.new_area("key", position=(-1, 2, -2), target=(0, 0, 0), strength=200)
    return scene


class HistorySamplesTests(unittest.TestCase):
    def test_gpu_cpu_parity_and_no_reconstruction_readback(self):
        results = []
        for cpu in (False, True):
            env = dict(os.environ)
            if cpu: env["PHOTARA_HISTORY_CPU"] = "1"
            else: env.pop("PHOTARA_HISTORY_CPU", None)
            with patch.dict(os.environ, env, clear=True), plane_scene(history_samples=2) as scene:
                images = []
                for i in range(6):
                    scene.camera.location = (0.015*i, 0, -4)
                    images.append(scene.render().beauty.copy())
                    stats = scene.history_stats()
                    self.assertLessEqual(stats["max_history_frames"], 2)
                self.assertEqual(stats["backend"], "CPU" if cpu else "GPU")
                if not cpu:
                    self.assertEqual(stats["gpu_updates"], 6)
                    self.assertEqual(stats["reconstruction_pixel_readback_bytes"], 0)
                else: self.assertGreater(stats["reconstruction_pixel_readback_bytes"], 0)
                results.append(images)
        np.testing.assert_allclose(results[0], results[1], rtol=3e-5, atol=3e-6)

    def test_default_adaptive_and_realtime_entry_points(self):
        for settings in ({"adaptive_sampling": True},
                         {"render_mode": "REALTIME", "samples_per_frame": 4}):
            with self.subTest(settings=settings), plane_scene(history_samples=2, **settings) as scene:
                scene.render()
                job = scene.render_async()
                self.assertTrue(np.isfinite(job.wait().beauty).all())
                self.assertEqual(scene.samples, 4)
                self.assertEqual(job.poll().num_samples, 4)
                self.assertEqual(scene.history_stats()["max_history_frames"], 1)

    def test_low_level_session_history_api(self):
        with pr.Session(48, 48, samples=2, history_samples=2, denoise=False) as session:
            self.assertEqual(session.history_samples, 2)
            session.render()
            session.render_async().wait()
            self.assertEqual(session.history_stats()["stored_frames"], 2)
            session.reset_history()
            self.assertEqual(session.history_stats()["stored_frames"], 0)
            session.history_samples = 0
            self.assertEqual(session.history_samples, 0)
        with self.assertRaises(pr.PhotaraClosedError): session.history_stats()

    def test_exact_window_uses_raw_frames_and_preserves_spp(self):
        seeds, outputs = [], []
        budgets = [4, 4, 4, 2, 6]
        with plane_scene(history_samples=2) as scene:
            self.assertEqual(scene.samples, 4)
            for i in range(5):
                scene.samples = budgets[i]
                job = scene.render_async()
                outputs.append(job.wait().beauty.copy())
                stats = scene.history_stats()
                seeds.append(stats["sample_seed"])
                self.assertEqual(job.poll().num_samples, budgets[i])
                self.assertEqual(stats["stored_frames"], min(i + 1, 2))
                self.assertEqual(stats["max_history_frames"], min(i, 2))
                if i: self.assertGreater(stats["mean_history_frames"], 0.99 * min(i, 2))
        raw = []
        with plane_scene() as scene:
            for seed, budget in zip(seeds, budgets):
                scene.samples = budget
                raw.append(scene.render(seed=seed).beauty.copy())
        # Independent single-frame renders verify expired frames cannot leak through
        # recursively averaged colors, and the current frame still contributes 4 SPP.
        for i, output in enumerate(outputs):
            start = max(0, i - 2)
            expected = np.average(raw[start:i + 1], axis=0, weights=budgets[start:i + 1])
            np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)

    def test_camera_reprojection_cut_mutation_and_resize(self):
        with plane_scene(history_samples=3) as scene:
            scene.render()
            scene.camera.location = (0.03, 0, -4)
            scene.render()
            self.assertGreater(scene.history_stats()["reused_pixels"], 2000)
            # Diagnostics must not clear history.
            scene.history_stats()
            scene._session._ctx.neeat_stats()
            scene.render()
            self.assertEqual(scene.history_stats()["max_history_frames"], 2)
            scene.reset_history()
            scene.render()
            self.assertEqual(scene.history_stats()["reused_pixels"], 0)
            scene.materials["wall"].base_color = (0.1, 0.2, 0.9)
            scene.render()
            self.assertEqual(scene.history_stats()["reused_pixels"], 0)
            scene.resize(48, 48)
            self.assertEqual(scene.render().beauty.shape, (48, 48, 4))
            self.assertEqual(scene.history_stats()["reused_pixels"], 0)

    def test_denoisers_receive_reconstructed_color(self):
        for denoiser in ("OPTIX", "OPENIMAGEDENOISE", "DLSS"):
            with self.subTest(denoiser=denoiser):
                with plane_scene(history_samples=2, denoise=True, denoiser=denoiser) as scene:
                    scene.render()
                    frame = scene.render()
                    self.assertTrue(np.isfinite(frame.beauty).all())
                    self.assertEqual(scene.history_stats()["max_history_frames"], 1)

    def test_disoccluded_pixels_do_not_reuse_occluder_color(self):
        with plane_scene(history_samples=2) as scene:
            mat = scene.materials.new("occluder", base_color=(0.02, 0.9, 0.02), roughness=1)
            mesh = scene.meshes.new("occluder", np.array([
                (-0.4, -1, -1), (-0.4, 1, -1), (0.4, 1, -1), (0.4, -1, -1)], np.float32),
                np.array([(0, 1, 2), (0, 2, 3)], np.int32), material=mat)
            scene.objects.new("occluder", mesh, object_id=2)
            previous = scene.render()
            scene.camera.location = (0.08, 0, -4)
            scene.camera.target = (0.08, 0, 0)
            current = scene.render()
            stats = scene.history_stats()
            self.assertGreater(stats["reused_pixels"], 1500)
            # Project current wall points into the old camera (parallel translation).
            focal = scene._session.height_ / (2 * np.tan(np.deg2rad(scene.camera.fov) / 2))
            old_x = np.floor(np.arange(64) + 0.5 + focal * 0.08 / 4).astype(int)
            valid_x = old_x < 64
            old_ids = previous.object_id[:, np.clip(old_x, 0, 63)]
            newly_visible = (current.object_id == 1) & (old_ids == 2) & valid_x[None, :]
            self.assertGreater(newly_visible.sum(), 10)
            scene.history_samples = 0
            raw = scene.render(seed=stats["sample_seed"])
            np.testing.assert_allclose(current.beauty[newly_visible], raw.beauty[newly_visible],
                                       rtol=2e-5, atol=2e-6)

    def test_disabled_reference_unchanged_and_validation(self):
        with plane_scene() as scene:
            reference = scene.render(seed=17).beauty.copy()
            for value in (-1, 65, 1.5, True, "2"):
                with self.assertRaises(ValueError): scene.history_samples = value
            self.assertEqual(scene.history_samples, 0)
            np.testing.assert_allclose(scene.render(seed=17).beauty, reference,
                                       rtol=1e-6, atol=1e-7)
            scene.history_samples = 2
            scene.render()
            scene.render()
            scene.history_samples = 0
            np.testing.assert_allclose(scene.render(seed=17).beauty, reference,
                                       rtol=1e-6, atol=1e-7)

    def test_nearly_coplanar_object_boundary_rejects_history(self):
        # Depth/normal tests alone cannot distinguish these two surfaces within
        # a pixel footprint. Reprojected neighboring pixels must match object ID.
        for cpu in (False, True):
            env = dict(os.environ)
            if cpu: env['PHOTARA_HISTORY_CPU'] = '1'
            else: env.pop('PHOTARA_HISTORY_CPU', None)
            with patch.dict(os.environ, env, clear=True), plane_scene(history_samples=2) as scene:
                mat = scene.materials.new('edge', base_color=(0.02, 0.9, 0.02), roughness=1)
                mesh = scene.meshes.new('edge', np.array([
                    (-0.4, -1, -0.0001), (-0.4, 1, -0.0001),
                    (0.4, 1, -0.0001), (0.4, -1, -0.0001)], np.float32),
                    np.array([(0, 1, 2), (0, 2, 3)], np.int32), material=mat)
                scene.objects.new('edge', mesh, object_id=2)
                previous = scene.render()
                scene.camera.location = (0.04, 0, -4)
                scene.camera.target = (0.04, 0, 0)
                current = scene.render()
                stats = scene.history_stats()
                focal = 48 / (2 * np.tan(np.deg2rad(scene.camera.fov) / 2))
                old_x = np.floor(np.arange(64) + 0.5 + focal * 0.04 / 4).astype(int)
                old_ids = previous.object_id[:, np.clip(old_x, 0, 63)]
                crossed = (current.object_id == 1) & (old_ids == 2) & (old_x < 64)[None, :]
                self.assertGreater(crossed.sum(), 0)
                scene.history_samples = 0
                raw = scene.render(seed=stats['sample_seed'])
                np.testing.assert_allclose(current.beauty[crossed], raw.beauty[crossed],
                                           rtol=2e-5, atol=2e-6)
    def test_running_job_rejects_history_mutations(self):
        with plane_scene(history_samples=2) as scene:
            scene.samples = 100000
            job = scene.render_async()
            for operation in (lambda: setattr(scene, "history_samples", 1), scene.reset_history,
                              scene.history_stats):
                with self.assertRaises(RuntimeError): operation()
            job.cancel()
            scene.samples = 4
            scene.render()
            self.assertEqual(scene.history_stats()["reused_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
