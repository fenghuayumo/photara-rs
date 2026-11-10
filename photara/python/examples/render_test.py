"""Minimal end-to-end test for the Photara Python renderer."""

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
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_outputs"))
    os.makedirs(out_dir, exist_ok=True)

    session = pr.Session(width=640, height=480, samples=32, device="OPTIX", denoise=True)
    print("device:", session.device_name)

    plane_material = session.add_material(
        pr.Material(base_color=(0.55, 0.12, 0.10), roughness=0.55)
    )
    sphere_material = session.add_material(
        pr.Material(base_color=(0.15, 0.55, 0.62), metallic=0.2, roughness=0.18)
    )

    plane_vertices = np.array(
        [(-8.0, 0.0, -8.0), (8.0, 0.0, -8.0), (8.0, 0.0, 8.0), (-8.0, 0.0, 8.0)],
        dtype=np.float32,
    )
    plane_indices = np.array([(0, 2, 1), (0, 3, 2)], dtype=np.int32)
    plane_normals = np.array([(0.0, 1.0, 0.0)] * 4, dtype=np.float32)
    plane_uvs = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float32)
    session.add_mesh(
        plane_vertices,
        plane_indices,
        normals=plane_normals,
        uvs=plane_uvs,
        material=plane_material,
        object_id=1,
    )

    sphere_vertices, sphere_indices, sphere_normals = uv_sphere(radius=1.0)
    sphere_transform = np.eye(4, dtype=np.float32)
    sphere_transform[1, 3] = 1.0
    session.add_mesh(
        sphere_vertices,
        sphere_indices,
        normals=sphere_normals,
        transform=sphere_transform,
        material=sphere_material,
        object_id=2,
    )

    session.set_camera(position=(4.2, 2.8, -5.5), target=(0.0, 1.0, 0.0), fov=42.0)
    session.add_area_light(
        position=(2.5, 4.5, -2.5),
        target=(0.0, 0.5, 0.0),
        size=(3.0, 3.0),
        color=(1.0, 0.95, 0.88),
        strength=300.0,
    )
    session.add_sun_light(direction=(-0.4, -1.0, 0.25), color=(1.0, 0.95, 0.9), strength=2.0, angle=0.5)
    session.set_background(color=(0.04, 0.07, 0.12), strength=1.0)

    start = time.perf_counter()
    session.render()
    elapsed = time.perf_counter() - start

    beauty = session.beauty()
    depth = session.depth()
    normal = session.normal()
    object_id = session.object_id()

    print(f"render time: {elapsed:.3f}s")
    print("beauty:", beauty.shape, beauty.dtype, "mean rgb:", beauty[..., :3].mean(axis=(0, 1)))
    print("depth:", depth.shape, "min/max:", float(depth.min()), float(depth.max()))
    print("normal:", normal.shape, "object id unique:", np.unique(object_id))

    pr.save_png(os.path.join(out_dir, "photara_test_beauty.png"), beauty)
    pr.save_png(os.path.join(out_dir, "photara_test_albedo.png"), session.albedo())
    pr.save_png(os.path.join(out_dir, "photara_test_normal.png"), session.normal() * 0.5 + 0.5)
    depth_vis = np.clip(depth, 0.0, 20.0) / 20.0
    depth_vis[depth > 1e6] = 0.0
    pr.save_png(os.path.join(out_dir, "photara_test_depth.png"), depth_vis[..., None], srgb=False)


if __name__ == "__main__":
    main()
