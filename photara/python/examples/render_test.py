"""Minimal end-to-end test for the Photara Scene API."""

from __future__ import annotations

import os
import time

import numpy as np

import photara_rs as pr


def uv_sphere(radius: float = 1.0, segments: int = 48, rings: int = 24):
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

    normals = vertices / radius
    return vertices, np.asarray(faces, dtype=np.int32), normals


def main() -> None:
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_outputs")
    )
    os.makedirs(out_dir, exist_ok=True)

    with pr.Scene(width=640, height=480, samples=32, device="OPTIX") as scene:
        print("device:", scene.device_name)

        floor_material = scene.materials.new(
            "floor", base_color=(0.55, 0.12, 0.10), roughness=0.55
        )
        sphere_material = scene.materials.new(
            "sphere", base_color=(0.15, 0.55, 0.62), metallic=0.2, roughness=0.18
        )

        plane_vertices = np.array(
            [(-8.0, 0.0, -8.0), (8.0, 0.0, -8.0), (8.0, 0.0, 8.0), (-8.0, 0.0, 8.0)],
            dtype=np.float32,
        )
        plane_indices = np.array([(0, 2, 1), (0, 3, 2)], dtype=np.int32)
        plane_normals = np.array([(0.0, 1.0, 0.0)] * 4, dtype=np.float32)
        plane_uvs = np.array(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float32
        )
        plane_mesh = scene.meshes.new(
            "plane",
            plane_vertices,
            plane_indices,
            normals=plane_normals,
            uvs=plane_uvs,
            material=floor_material,
        )
        scene.objects.new("plane", plane_mesh, object_id=1)

        sphere_vertices, sphere_indices, sphere_normals = uv_sphere(radius=1.0)
        sphere_mesh = scene.meshes.new(
            "sphere", sphere_vertices, sphere_indices, normals=sphere_normals,
            material=sphere_material,
        )
        scene.objects.new(
            "sphere",
            sphere_mesh,
            location=(0.0, 1.0, 0.0),
            object_id=2,
        )
        # A second object sharing the same mesh exercises instancing.
        scene.objects.new(
            "sphere.001",
            sphere_mesh,
            location=(-2.0, 1.0, 0.5),
            object_id=3,
        )

        scene.camera.location = (4.2, 2.8, -5.5)
        scene.camera.target = (0.0, 1.0, 0.0)
        scene.camera.fov = 42.0

        scene.lights.new_area(
            "key",
            position=(2.5, 4.5, -2.5),
            target=(0.0, 0.5, 0.0),
            size=(3.0, 3.0),
            color=(1.0, 0.95, 0.88),
            strength=300.0,
        )
        scene.lights.new_sun(
            "sun",
            direction=(-0.4, -1.0, 0.25),
            color=(1.0, 0.95, 0.9),
            strength=2.0,
            angle=0.5,
        )
        scene.world.color = (0.04, 0.07, 0.12)
        scene.world.strength = 1.0

        start = time.perf_counter()
        result = scene.render(seed=1)
        elapsed = time.perf_counter() - start

        print(f"render time: {elapsed:.3f}s")
        print("beauty:", result.beauty.shape, result.beauty.dtype,
              "mean rgb:", result.rgb.mean(axis=(0, 1)))
        print("depth:", result.depth.shape, "valid:", result.depth_valid.mean())
        print("normal:", result.normal.shape,
              "object ids:", np.unique(result.object_id))

        pr.save_png(os.path.join(out_dir, "photara_test_beauty.png"), result.beauty)
        pr.save_png(os.path.join(out_dir, "photara_test_albedo.png"), result.albedo)
        pr.save_png(
            os.path.join(out_dir, "photara_test_normal.png"),
            result.normal * 0.5 + 0.5,
        )
        depth_vis = np.clip(result.depth, 0.0, 20.0) / 20.0
        depth_vis[~result.depth_valid] = 0.0
        pr.save_png(
            os.path.join(out_dir, "photara_test_depth.png"),
            depth_vis[..., None],
            srgb=False,
        )


if __name__ == "__main__":
    main()
