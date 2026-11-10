"""End-to-end tests for the public Scene/Object API.

Run from the repository root with:

    set PYTHONPATH=%CD%\\photara\\python
    python -m unittest discover -s photara\\python\\tests -v

The tests require an OptiX-capable device and are skipped otherwise.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import photara_rs as pr


def uv_sphere(radius: float = 1.0, segments: int = 16, rings: int = 8):
    vertices = []
    for iy in range(rings + 1):
        phi = (iy / rings) * np.pi
        for ix in range(segments):
            theta = (ix / segments) * 2.0 * np.pi
            vertices.append(
                (
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.cos(phi),
                    radius * np.sin(phi) * np.sin(theta),
                )
            )
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = []
    for iy in range(rings):
        for ix in range(segments):
            i0 = iy * segments + ix
            i1 = iy * segments + (ix + 1) % segments
            i2 = (iy + 1) * segments + ix
            i3 = (iy + 1) * segments + (ix + 1) % segments
            faces.append((i0, i1, i2))
            faces.append((i1, i3, i2))
    return vertices, np.asarray(faces, dtype=np.int32), vertices / radius


def build_scene():
    scene = pr.Scene(
        width=64,
        height=48,
        samples=2,
        max_bounces=2,
        denoise=False,
        adaptive_sampling=False,
        device="OPTIX",
    )
    material = scene.materials.new("prop", base_color=(0.8, 0.2, 0.1), roughness=0.4)
    floor_vertices = np.array(
        [(-8.0, 0.0, -8.0), (8.0, 0.0, -8.0), (8.0, 0.0, 8.0), (-8.0, 0.0, 8.0)],
        dtype=np.float32,
    )
    floor_indices = np.array([(0, 2, 1), (0, 3, 2)], dtype=np.int32)
    floor_normals = np.array([(0.0, 1.0, 0.0)] * 4, dtype=np.float32)
    floor_mesh = scene.meshes.new(
        "floor", floor_vertices, floor_indices, normals=floor_normals,
        material=material,
    )
    scene.objects.new("floor", floor_mesh, object_id=1)

    vertices, indices, normals = uv_sphere()
    ball_mesh = scene.meshes.new(
        "ball", vertices, indices, normals=normals, material=material
    )
    ball = scene.objects.new("ball", ball_mesh, location=(0.0, 1.0, 0.0), object_id=2)
    instance = scene.objects.new(
        "ball.001", ball_mesh, location=(-1.5, 1.0, 0.0), object_id=3
    )

    scene.camera.location = (4.0, 3.0, -5.0)
    scene.camera.target = (0.0, 1.0, 0.0)
    scene.camera.fov = 45.0
    scene.lights.new_area(
        "key",
        position=(2.0, 4.0, -2.0),
        target=(0.0, 0.5, 0.0),
        size=(2.0, 2.0),
        strength=300.0,
    )
    scene.world.color = (0.04, 0.07, 0.12)
    return scene, material, ball_mesh, ball, instance, floor_mesh


class SceneApiTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            (
                self.scene,
                self.material,
                self.ball_mesh,
                self.ball,
                self.instance,
                self.floor_mesh,
            ) = build_scene()
        except RuntimeError as exc:
            self.skipTest(f"OptiX is unavailable: {exc}")

    def tearDown(self) -> None:
        if getattr(self, "scene", None) is not None:
            self.scene.close()

    def test_passes_and_npz(self) -> None:
        result = self.scene.render(seed=1)
        self.assertEqual(result.beauty.shape, (48, 64, 4))
        self.assertEqual(result.beauty.dtype, np.float32)
        self.assertEqual(result.depth.shape, (48, 64))
        self.assertEqual(result.depth_valid.dtype, np.bool_)
        self.assertEqual(result.normal.shape, (48, 64, 3))
        self.assertEqual(result.albedo.shape, (48, 64, 3))
        self.assertEqual(result.object_id.dtype, np.int32)
        self.assertIn(2, np.unique(result.object_id))
        self.assertTrue(result.depth_valid.any())

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "frame.npz")
            result.save_npz(path)
            with np.load(path) as data:
                self.assertEqual(data["combined"].shape, (48, 64, 4))
                self.assertEqual(data["object_id"].dtype, np.int32)
                self.assertEqual(int(data["seed"]), 1)

    def test_material_and_object_updates(self) -> None:
        first = self.scene.render(seed=1)
        first_mean = float(first.rgb.mean())

        self.material.base_color = (0.1, 0.5, 0.9)
        self.material.update(roughness=0.15, metallic=0.2)
        self.ball.location = (0.5, 1.2, 0.0)
        self.ball.rotation_euler = (0.2, 0.4, 0.0)
        self.ball.object_id = 7
        self.instance.visible = False
        self.scene.lights["key"].strength = 500.0

        second = self.scene.render(seed=2)
        self.assertGreater(float(np.abs(first.beauty - second.beauty).mean()), 1e-3)
        self.assertIn(7, np.unique(second.object_id))
        self.assertNotIn(3, np.unique(second.object_id))
        self.assertNotAlmostEqual(first_mean, float(second.rgb.mean()), places=3)

    def test_mesh_deformation_and_topology(self) -> None:
        before = self.scene.render(seed=20)
        vertices = self.ball_mesh.vertices.copy()
        deformed = vertices.copy()
        deformed[:, 0] *= 0.8
        deformed[:, 1] *= 1.3

        self.ball_mesh.update_positions(deformed)
        after_deform = self.scene.render(seed=21)
        self.assertGreater(
            float(np.abs(before.beauty - after_deform.beauty).mean()), 1e-4
        )
        self.assertEqual(self.ball_mesh.users, 2)

        normals = deformed / np.linalg.norm(deformed, axis=1, keepdims=True)
        uvs = np.zeros((deformed.shape[0], 2), dtype=np.float32)
        self.ball_mesh.update_positions(deformed, normals=normals, uvs=uvs)
        self.scene.render(seed=22)

        new_vertices = np.array(
            [(-1.0, 0.0, -1.0), (1.0, 0.0, -1.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0)],
            dtype=np.float32,
        )
        new_indices = np.array(
            [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)], dtype=np.int32
        )
        material_before = self.ball_mesh.material
        self.ball_mesh.update_topology(new_vertices, new_indices)
        self.assertEqual(self.ball_mesh.vertices.shape, (4, 3))
        self.assertEqual(self.ball_mesh.indices.shape, (4, 3))
        self.assertEqual(self.ball_mesh.users, 2)
        self.assertIs(self.ball_mesh.material, material_before)

        after_topology = self.scene.render(seed=23)
        self.assertIn(2, np.unique(after_topology.object_id))
        self.assertIn(3, np.unique(after_topology.object_id))

        with self.assertRaises(RuntimeError):
            self.ball_mesh.update_positions(np.zeros((3, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            self.ball_mesh.update_topology(
                new_vertices, np.array([(0, 1, 9)], dtype=np.int32)
            )

    def test_instances_and_removal(self) -> None:
        self.assertEqual(self.ball_mesh.users, 2)
        self.scene.objects.remove(self.instance)
        self.assertEqual(self.ball_mesh.users, 1)
        self.assertNotIn("ball.001", self.scene.objects)

        self.scene.meshes.remove(self.ball_mesh)
        self.assertNotIn("ball", self.scene.objects)
        self.assertIn("floor", self.scene.objects)
        with self.assertRaises(pr.PhotaraRemovedError):
            _ = self.ball.location

    def test_point_and_spot_lights(self) -> None:
        first = self.scene.render(seed=10)
        point = self.scene.lights.new_point(
            "point", position=(0.0, 2.0, -1.0), strength=200.0, radius=0.0
        )
        self.assertEqual(point.light_type, "POINT")
        second = self.scene.render(seed=11)
        self.assertGreater(float(np.abs(first.beauty - second.beauty).mean()), 1e-4)

        point.location = (1.0, 2.5, -1.0)
        point.strength = 300.0
        point.radius = 0.5
        self.assertEqual(point.radius, 0.5)
        third = self.scene.render(seed=12)
        self.assertGreater(float(np.abs(second.beauty - third.beauty).mean()), 1e-4)

        spot = self.scene.lights.new_spot(
            "spot",
            position=(0.0, 3.0, -1.0),
            direction=(0.0, -1.0, 0.2),
            angle=30.0,
            smooth=0.0,
        )
        self.assertEqual(spot.light_type, "SPOT")
        fourth = self.scene.render(seed=13)
        self.assertGreater(float(np.abs(third.beauty - fourth.beauty).mean()), 1e-4)
        spot.spot_angle = 20.0
        spot.direction = (0.2, -1.0, 0.1)
        self.scene.render(seed=14)

        with self.assertRaises(ValueError):
            self.scene.lights.new_point("bad_point", radius=-1.0)
        with self.assertRaises(ValueError):
            self.scene.lights.new_spot("bad_spot", angle=0.0)
        with self.assertRaises(RuntimeError):
            self.scene.lights.new_spot("bad_spot", direction=(0.0, 0.0, 0.0))

    def test_camera_and_light_editing(self) -> None:
        camera = self.scene.camera
        camera.location = (3.0, 2.0, -5.0)
        camera.target = (0.0, 0.5, 0.0)
        camera.fov = 50.0
        self.assertAlmostEqual(camera.fov, 50.0)

        light = self.scene.lights["key"]
        light.location = (1.0, 3.0, -1.0)
        light.target = (0.0, 0.0, 0.0)
        light.size = (1.5, 1.0)
        light.color = (0.8, 0.9, 1.0)
        light.strength = 250.0
        sun = self.scene.lights.new_sun("sun", direction=(-0.3, -1.0, 0.2))
        sun.direction = (-0.2, -1.0, 0.1)
        sun.angle = 0.4
        self.scene.world.strength = 0.8

        result = self.scene.render(seed=3)
        self.assertEqual(result.object_id.shape, (48, 64))

    def test_validation_removed_and_closed(self) -> None:
        with self.assertRaises(ValueError):
            pr.Material(metallic=2.0)
        with self.assertRaises(ValueError):
            pr.Material(base_color=(-1.0, 0.0, 0.0))
        with self.assertRaises(TypeError):
            self.scene.objects.new(
                "bad", self.ball_mesh, transform=np.eye(4), location=(0, 0, 0)
            )
        with self.assertRaises(ValueError):
            self.scene.objects.new("bad_id", self.ball_mesh, object_id=2**24 + 1)
        with self.assertRaises(TypeError):
            self.scene.meshes.new(
                "bad", np.zeros((3, 3), np.float32), np.array([[0.0, 1.0, 2.0]])
            )
        with self.assertRaises(ValueError):
            self.scene.meshes.new(
                "bad", np.zeros((3, 3), np.float32), np.array([[0, 1, 9]], np.int32)
            )
        with self.assertRaises(pr.PhotaraError):
            self.scene.materials.remove(self.material)

        unused = self.scene.materials.new("unused")
        self.scene.materials.remove(unused)
        self.assertFalse(unused.bound)

        self.scene.close()
        self.assertTrue(self.scene.closed)
        with self.assertRaises(pr.PhotaraClosedError):
            self.scene.render()
        with self.assertRaises(pr.PhotaraClosedError):
            _ = self.ball.location

    def test_low_level_session(self) -> None:
        with pr.Session(
            width=32, height=24, samples=1, denoise=False, adaptive_sampling=False
        ) as session:
            material = pr.Material(base_color=(0.8, 0.2, 0.1), metallic=0.3)
            index = session.add_material(material)
            session.update_material(index, roughness=0.2)
            self.assertAlmostEqual(material.roughness, 0.2)
            self.assertAlmostEqual(material.metallic, 0.3)

            vertices = np.array(
                [(-2.0, 0.0, -2.0), (2.0, 0.0, -2.0), (0.0, 2.0, 0.0)],
                dtype=np.float32,
            )
            indices = np.array([(0, 1, 2)], dtype=np.int32)
            handle = session.add_mesh(vertices, indices, material=index, object_id=1)
            session.set_camera(position=(3.0, 2.0, -4.0), target=(0.0, 0.5, 0.0))
            session.add_area_light(
                position=(1.0, 3.0, -1.0), target=(0.0, 0.0, 0.0), strength=100.0
            )
            session.render(seed=1)
            self.assertIn(1.0, np.unique(session.object_id()))
            session.remove_object(handle)

    def test_srgb_alpha(self) -> None:
        rgba = np.array([[[0.5, 0.5, 0.5, 0.25]]], dtype=np.float32)
        converted = pr.linear_to_srgb(rgba)
        self.assertAlmostEqual(float(converted[0, 0, 3]), 0.25, places=6)
        self.assertGreater(float(converted[0, 0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
