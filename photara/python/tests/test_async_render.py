"""Async render/progress tests for the public Python API."""

from __future__ import annotations

import time
import unittest

import numpy as np

import photara_rs as pr


def build_scene(width=64, height=48, samples=2):
    scene = pr.Scene(
        width,
        height,
        samples=samples,
        max_bounces=2,
        denoise=False,
        adaptive_sampling=False,
        device="OPTIX",
    )
    material = scene.materials.new("mat", base_color=(0.7, 0.2, 0.1))
    vertices = np.array(
        [(-3.0, 0.0, -3.0), (3.0, 0.0, -3.0), (0.0, 1.5, 0.0)],
        dtype=np.float32,
    )
    indices = np.array([(0, 1, 2)], dtype=np.int32)
    mesh = scene.meshes.new("mesh", vertices, indices, material=material)
    obj = scene.objects.new("mesh", mesh, object_id=2)
    scene.camera.location = (3.0, 2.0, -4.0)
    scene.camera.target = (0.0, 0.5, 0.0)
    scene.lights.new_area(
        "key", position=(1.0, 3.0, -1.0), target=(0.0, 0.0, 0.0), strength=200.0
    )
    return scene, obj


class AsyncRenderTests(unittest.TestCase):
    def test_progress_and_wait(self) -> None:
        try:
            scene, _ = build_scene(samples=8)
        except RuntimeError as exc:
            self.skipTest(f"OptiX is unavailable: {exc}")
        try:
            job = scene.render_async(seed=1)
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                snapshot = job.poll()
                if snapshot.state != "running":
                    break
                time.sleep(0.005)
            result = job.wait()
            self.assertEqual(result.beauty.shape, (48, 64, 4))
            self.assertIn(2, np.unique(result.object_id))
            snapshot = job.poll()
            self.assertEqual(snapshot.state, "done")
            self.assertEqual(snapshot.num_samples, 8)
            self.assertGreater(snapshot.current_sample, 0)
            self.assertAlmostEqual(snapshot.progress, 1.0, places=6)
        finally:
            scene.close()

    def test_mutation_guard_and_cancel(self) -> None:
        try:
            scene, obj = build_scene(width=512, height=384, samples=4096)
        except RuntimeError as exc:
            self.skipTest(f"OptiX is unavailable: {exc}")
        try:
            job = scene.render_async(seed=2)
            with self.assertRaises(RuntimeError) as context:
                obj.location = (1.0, 0.0, 0.0)
            self.assertIn("render is running", str(context.exception))

            job.cancel()
            self.assertEqual(job.state, "cancelled")
            with self.assertRaises(pr.PhotaraCancelledError):
                job.wait()

            scene.samples = 4
            result = scene.render(seed=3)
            self.assertEqual(result.beauty.shape, (384, 512, 4))
        finally:
            scene.close()

    def test_requested_cancel_resize_and_preview(self):
        scene, obj = build_scene(width=128, height=96, samples=4096)
        try:
            job = scene.render_async()
            job.request_cancel()
            deadline = time.monotonic() + 15
            while job.poll().state == "running":
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.001)
            self.assertIn(job.state, ("cancelled", "done"))
            scene.samples = 2
            for width, height in ((64, 48), (128, 96)):
                scene.resize(width, height)
                job = scene.render_async()
                result = job.wait()
                preview = job.preview()
                self.assertEqual(preview.beauty.shape, (height, width, 4))
                np.testing.assert_array_equal(preview.beauty, result.beauty)
                np.testing.assert_array_equal(preview.depth_valid, result.depth_valid)
                self.assertIn(2, result.object_id)
            with self.assertRaises(ValueError):
                scene.resize(0, 48)
        finally:
            scene.close()

    def test_native_progressive_updates_share_one_job(self):
        with pr.Scene(640, 480, samples=2048, denoise=False,
                      adaptive_sampling=False, viewport=True) as scene:
            job = scene.render_async()
            revision = 0
            running_updates = []
            deadline = time.monotonic() + 15
            while True:
                progress = job.poll()
                update = job.preview_update(revision)
                if update is not None:
                    new_revision, frame = update
                    self.assertGreater(new_revision, revision)
                    revision = new_revision
                    self.assertEqual(frame.beauty.shape, (480, 640, 4))
                    if progress.state == 'running':
                        running_updates.append(progress.current_sample)
                if progress.state != 'running':
                    self.assertEqual(progress.state, 'done')
                    break
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.005)
            self.assertGreaterEqual(len(set(running_updates)), 2)
            self.assertEqual(job.id, 1)

    def test_close_while_running(self) -> None:
        try:
            scene, _ = build_scene(width=512, height=384, samples=4096)
        except RuntimeError as exc:
            self.skipTest(f"OptiX is unavailable: {exc}")
        job = scene.render_async(seed=4)
        scene.close()
        self.assertTrue(scene.closed)
        with self.assertRaises(pr.PhotaraClosedError):
            job.poll()

    def test_concurrent_contexts(self) -> None:
        try:
            scene_a, _ = build_scene()
            scene_b, _ = build_scene()
        except RuntimeError as exc:
            self.skipTest(f"OptiX is unavailable: {exc}")
        try:
            job_a = scene_a.render_async(seed=10)
            job_b = scene_b.render_async(seed=11)
            result_a = job_a.wait()
            result_b = job_b.wait()
            self.assertEqual(result_a.beauty.shape, (48, 64, 4))
            self.assertEqual(result_b.beauty.shape, (48, 64, 4))
            self.assertEqual(job_a.poll().state, "done")
            self.assertEqual(job_b.poll().state, "done")
        finally:
            scene_a.close()
            scene_b.close()


if __name__ == "__main__":
    unittest.main()
