"""Texture and environment-map tests for the Scene API."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import photara_rs as pr

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is an optional extra.
    Image = None


def _write_split_texture(path: str) -> None:
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[:, :32] = (255, 0, 0)
    image[:, 32:] = (0, 0, 255)
    Image.fromarray(image).save(path)


def _write_environment(path: str) -> None:
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[:, :32] = (255, 160, 60)
    image[:, 32:] = (40, 80, 255)
    Image.fromarray(image).save(path)


class TextureTests(unittest.TestCase):
    def setUp(self) -> None:
        if Image is None:
            self.skipTest("Pillow is not installed")
        self._tmp = tempfile.TemporaryDirectory()
        self.texture_path = os.path.join(self._tmp.name, "split.png")
        self.environment_path = os.path.join(self._tmp.name, "env.png")
        _write_split_texture(self.texture_path)
        _write_environment(self.environment_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _plane_scene(self):
        scene = pr.Scene(
            128,
            96,
            samples=4,
            max_bounces=2,
            denoise=False,
            adaptive_sampling=False,
            device="OPTIX",
        )
        material = scene.materials.new(
            "textured", base_color=(1.0, 1.0, 1.0), roughness=0.5
        )
        vertices = np.array(
            [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
            dtype=np.float32,
        )
        indices = np.array([(0, 1, 2), (0, 2, 3)], dtype=np.int32)
        normals = np.array([(0.0, 0.0, 1.0)] * 4, dtype=np.float32)
        uvs = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float32)
        mesh = scene.meshes.new(
            "plane", vertices, indices, normals=normals, uvs=uvs, material=material
        )
        scene.objects.new("plane", mesh, object_id=1)
        scene.camera.location = (0.0, 0.0, 3.0)
        scene.camera.target = (0.0, 0.0, 0.0)
        scene.lights.new_area(
            "key",
            position=(0.0, 0.0, 2.0),
            target=(0.0, 0.0, 0.0),
            size=(2.0, 2.0),
            strength=80.0,
        )
        scene.world.color = (0.01, 0.01, 0.01)
        return scene, material

    def test_base_color_texture(self) -> None:
        try:
            scene, material = self._plane_scene()
        except RuntimeError as exc:
            self.skipTest(f"OptiX is unavailable: {exc}")
        try:
            scene.render(seed=1)
            material.base_color_texture = {
                "path": self.texture_path,
                "colorspace": "sRGB",
                "interpolation": "linear",
                "extension": "repeat",
            }
            result = scene.render(seed=2)
            left = result.albedo[20:76, 32:64].mean(axis=(0, 1))
            right = result.albedo[20:76, 64:96].mean(axis=(0, 1))
            self.assertGreater(abs(float(left[0]) - float(right[0])), 0.1)
            self.assertGreater(abs(float(left[2]) - float(right[2])), 0.1)

            material.base_color_texture = None
            cleared = scene.render(seed=3)
            self.assertLess(float(cleared.albedo[20:76, 32:96].std()), 0.05)

            with self.assertRaises(FileNotFoundError):
                material.base_color_texture = {"path": os.path.join(self._tmp.name, "missing.png")}
            with self.assertRaises(ValueError):
                material.base_color_texture = {
                    "path": self.texture_path,
                    "colorspace": "not-a-colorspace",
                }
        finally:
            scene.close()

    def test_environment_texture(self) -> None:
        try:
            scene = pr.Scene(
                96,
                64,
                samples=2,
                max_bounces=1,
                denoise=False,
                adaptive_sampling=False,
                device="OPTIX",
            )
        except RuntimeError as exc:
            self.skipTest(f"OptiX is unavailable: {exc}")
        try:
            scene.camera.location = (0.0, 0.0, -3.0)
            scene.camera.target = (0.0, 0.0, 0.0)
            plain = scene.render(seed=1)
            self.assertLess(float(plain.rgb.std()), 0.01)

            scene.world.environment_texture = {
                "path": self.environment_path,
                "colorspace": "sRGB",
            }
            lit = scene.render(seed=2)
            self.assertGreater(float(lit.rgb.std()), 0.05)
            self.assertGreater(float(lit.rgb.mean()), 0.01)

            scene.world.environment_texture = None
            restored = scene.render(seed=3)
            self.assertLess(float(restored.rgb.std()), 0.01)
            self.assertAlmostEqual(
                float(restored.rgb.mean()), float(plain.rgb.mean()), places=4
            )
        finally:
            scene.close()

    def test_environment_before_first_render(self) -> None:
        try:
            scene = pr.Scene(
                64,
                48,
                samples=1,
                max_bounces=1,
                denoise=False,
                adaptive_sampling=False,
                device="OPTIX",
            )
        except RuntimeError as exc:
            self.skipTest(f"OptiX is unavailable: {exc}")
        try:
            scene.camera.location = (0.0, 0.0, -3.0)
            scene.camera.target = (0.0, 0.0, 0.0)
            scene.world.environment_texture = self.environment_path
            result = scene.render(seed=4)
            self.assertGreater(float(result.rgb.std()), 0.03)
        finally:
            scene.close()


if __name__ == "__main__":
    unittest.main()
