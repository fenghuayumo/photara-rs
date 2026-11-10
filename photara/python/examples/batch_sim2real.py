"""Batch sim2real rendering example using the Scene/Object API.

Builds one scene, then reuses it for several frames while updating object
transforms, camera pose, material parameters and lights. Each frame is written
as a compressed npz file with raw float32 passes, which is the format most
simulation pipelines consume directly.

Run from the repository root with the development package on PYTHONPATH:

    set PYTHONPATH=%CD%\\photara\\python
    python photara\\python\\examples\\batch_sim2real.py --frames 4

Outputs go to test_outputs/batch_sim2real by default.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import photara_rs as pr


def rotation_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = c
    matrix[0, 2] = s
    matrix[2, 0] = -s
    matrix[2, 2] = c
    return matrix


def uv_sphere(radius: float = 1.0, segments: int = 32, rings: int = 16):
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


def build_scene(scene: pr.Scene):
    floor_material = scene.materials.new(
        "floor", base_color=(0.35, 0.35, 0.38), roughness=0.65
    )
    prop_material = scene.materials.new(
        "prop", base_color=(0.8, 0.25, 0.08), metallic=0.1, roughness=0.3
    )

    floor_vertices = np.array(
        [(-10.0, 0.0, -10.0), (10.0, 0.0, -10.0), (10.0, 0.0, 10.0), (-10.0, 0.0, 10.0)],
        dtype=np.float32,
    )
    floor_indices = np.array([(0, 2, 1), (0, 3, 2)], dtype=np.int32)
    floor_normals = np.array([(0.0, 1.0, 0.0)] * 4, dtype=np.float32)
    floor_mesh = scene.meshes.new(
        "floor", floor_vertices, floor_indices, normals=floor_normals,
        material=floor_material,
    )
    scene.objects.new("floor", floor_mesh, object_id=1)

    sphere_vertices, sphere_indices, sphere_normals = uv_sphere()
    sphere_mesh = scene.meshes.new(
        "sphere", sphere_vertices, sphere_indices, normals=sphere_normals,
        material=prop_material,
    )
    source = scene.objects.new(
        "sphere", sphere_mesh, location=(0.0, 1.0, 0.0), object_id=2
    )
    instances = [
        scene.objects.new(
            "sphere.001", sphere_mesh, location=(-2.5, 1.0, 0.0), object_id=3
        ),
        scene.objects.new(
            "sphere.002", sphere_mesh, location=(2.5, 1.0, 0.0), object_id=4
        ),
    ]

    scene.camera.location = (5.0, 3.0, -6.0)
    scene.camera.target = (0.0, 1.0, 0.0)
    scene.camera.fov = 48.0

    area_light = scene.lights.new_area(
        "key",
        position=(2.5, 5.0, -2.0),
        target=(0.0, 0.5, 0.0),
        size=(3.0, 3.0),
        color=(1.0, 0.95, 0.88),
        strength=400.0,
    )
    sun_light = scene.lights.new_sun(
        "sun",
        direction=(-0.4, -1.0, 0.25),
        color=(1.0, 0.95, 0.9),
        strength=1.5,
        angle=0.5,
    )
    scene.world.color = (0.04, 0.06, 0.10)
    scene.world.strength = 1.0
    return prop_material, source, instances, area_light, sun_light


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--output",
        default=os.path.join("test_outputs", "batch_sim2real"),
    )
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with pr.Scene(
        width=args.width,
        height=args.height,
        samples=args.samples,
        max_bounces=4,
        denoise=True,
        adaptive_sampling=True,
        device="OPTIX",
    ) as scene:
        print("scene:", scene)
        prop_material, source, instances, area_light, sun_light = build_scene(scene)

        for frame in range(args.frames):
            angle = 0.35 * frame

            source.matrix_world = rotation_y(angle) @ source.matrix_world
            source.location = (0.0, 1.0, 0.0)

            for index, instance in enumerate(instances):
                transform = rotation_y(-angle * 1.7)
                transform[0, 3] = -2.5 if index == 0 else 2.5
                transform[1, 3] = 1.0 + 0.25 * np.sin(angle + index)
                instance.matrix_world = transform

            # Simulated camera trajectory (replace with sensor poses from the sim).
            scene.camera.location = (
                5.0 * np.cos(angle * 0.5),
                3.0,
                -6.0 * np.sin(angle * 0.5 + 0.5),
            )
            scene.camera.target = (0.0, 1.0, 0.0)
            scene.camera.fov = 48.0

            # Domain randomization: material and light variation per frame.
            prop_material.update(
                base_color=(
                    0.5 + 0.4 * np.sin(angle),
                    0.25 + 0.2 * np.cos(angle * 1.3),
                    0.1 + 0.15 * np.sin(angle * 0.7),
                ),
                metallic=0.1,
                roughness=0.25 + 0.15 * np.sin(angle),
            )
            area_light.location = (
                2.5 * np.cos(angle),
                5.0,
                -2.0 * np.sin(angle),
            )
            area_light.target = (0.0, 0.5, 0.0)
            area_light.color = (1.0, 0.9 + 0.1 * np.sin(angle), 0.85)
            area_light.strength = 350.0 + 100.0 * np.sin(angle)
            sun_light.direction = (-0.4, -1.0, 0.25 + 0.15 * np.cos(angle))

            seed = 1000 + frame
            start = time.perf_counter()
            result = scene.render(seed=seed)
            elapsed = time.perf_counter() - start

            result.save_npz(os.path.join(args.output, f"frame_{frame:04d}.npz"))
            if not args.no_preview:
                result.save_png(os.path.join(args.output, f"frame_{frame:04d}.png"))

            print(
                f"frame {frame:04d}: {elapsed:.3f}s seed={seed} "
                f"objects={np.unique(result.object_id).tolist()}"
            )

    print("wrote", args.output)


if __name__ == "__main__":
    main()
