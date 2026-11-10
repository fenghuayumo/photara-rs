"""Batch sim2real rendering example.

Builds one scene, then reuses the same OptiX session for several frames while
updating object transforms, camera pose, material parameters and lights. Each
frame is written as a compressed npz file with raw float32 passes, which is the
format most simulation pipelines want to consume directly.

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
    """Return a 4x4 Y-rotation matrix for column vectors."""
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = c
    matrix[0, 2] = s
    matrix[2, 0] = -s
    matrix[2, 2] = c
    return matrix


def camera_matrix(
    position,
    target,
    up=(0.0, 1.0, 0.0),
) -> np.ndarray:
    """Build a camera transform: local +Z looks at target, +Y is up."""
    position = np.asarray(position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    forward = target - position
    forward /= np.linalg.norm(forward)
    hint = np.asarray(up, dtype=np.float32)
    right = np.cross(hint, forward)
    right /= np.linalg.norm(right)
    up_vector = np.cross(forward, right)

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 0] = right
    matrix[:3, 1] = up_vector
    matrix[:3, 2] = forward
    matrix[:3, 3] = position
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


def build_scene(session: pr.Session):
    """Create the static part of the scene; returns handles to mutate later."""
    floor_material = session.add_material(
        pr.Material(base_color=(0.35, 0.35, 0.38), roughness=0.65)
    )
    prop_material = session.add_material(
        pr.Material(base_color=(0.8, 0.25, 0.08), metallic=0.1, roughness=0.3)
    )

    floor_vertices = np.array(
        [(-10.0, 0.0, -10.0), (10.0, 0.0, -10.0), (10.0, 0.0, 10.0), (-10.0, 0.0, 10.0)],
        dtype=np.float32,
    )
    floor_indices = np.array([(0, 2, 1), (0, 3, 2)], dtype=np.int32)
    floor_normals = np.array([(0.0, 1.0, 0.0)] * 4, dtype=np.float32)
    session.add_mesh(
        floor_vertices,
        floor_indices,
        normals=floor_normals,
        material=floor_material,
        object_id=1,
    )

    sphere_vertices, sphere_indices, sphere_normals = uv_sphere()
    source = session.add_mesh(
        sphere_vertices,
        sphere_indices,
        normals=sphere_normals,
        material=prop_material,
        object_id=2,
    )
    instances = []
    for index, x in enumerate((-2.5, 2.5)):
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = x
        transform[1, 3] = 1.0
        instances.append(
            session.add_instance(source, transform, object_id=3 + index)
        )

    session.set_camera(position=(5.0, 3.0, -6.0), target=(0.0, 1.0, 0.0), fov=48.0)
    area_light = session.add_area_light(
        position=(2.5, 5.0, -2.0),
        target=(0.0, 0.5, 0.0),
        size=(3.0, 3.0),
        color=(1.0, 0.95, 0.88),
        strength=400.0,
    )
    sun_light = session.add_sun_light(
        direction=(-0.4, -1.0, 0.25),
        color=(1.0, 0.95, 0.9),
        strength=1.5,
        angle=0.5,
    )
    session.set_background(color=(0.04, 0.06, 0.10), strength=1.0)
    return prop_material, source, instances, area_light, sun_light


def save_passes(session: pr.Session, output_dir: str, frame: int, seed: int) -> None:
    depth = session.depth()
    passes = {
        "beauty": session.beauty(),
        "depth": depth,
        "depth_valid": depth < 1e9,
        "normal": session.normal(),
        "albedo": session.albedo(),
        "object_id": session.object_id(),
        "seed": np.int32(seed),
    }
    np.savez_compressed(os.path.join(output_dir, f"frame_{frame:04d}.npz"), **passes)


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

    session = pr.Session(
        width=args.width,
        height=args.height,
        samples=args.samples,
        max_bounces=4,
        denoise=True,
        adaptive_sampling=True,
        device="OPTIX",
    )
    print("device:", session.device_name)

    prop_material, source, instances, area_light, sun_light = build_scene(session)

    for frame in range(args.frames):
        angle = 0.35 * frame

        # Animate the source ball and its instances.
        source_transform = rotation_y(angle)
        source_transform[1, 3] = 1.0
        session.set_object_transform(source, source_transform)

        for index, instance in enumerate(instances):
            transform = rotation_y(-angle * 1.7)
            transform[0, 3] = -2.5 if index == 0 else 2.5
            transform[1, 3] = 1.0 + 0.25 * np.sin(angle + index)
            session.set_object_transform(instance, transform)

        # Simulated camera pose (for example from a robot/sensor trajectory).
        camera_position = (
            5.0 * np.cos(angle * 0.5),
            3.0,
            -6.0 * np.sin(angle * 0.5 + 0.5),
        )
        session.set_camera_matrix(
            camera_matrix(camera_position, target=(0.0, 1.0, 0.0)),
            fov=48.0,
        )

        # Domain randomization: material and light variation per frame.
        session.update_material(
            prop_material,
            base_color=(
                0.5 + 0.4 * np.sin(angle),
                0.25 + 0.2 * np.cos(angle * 1.3),
                0.1 + 0.15 * np.sin(angle * 0.7),
            ),
            metallic=0.1,
            roughness=0.25 + 0.15 * np.sin(angle),
        )
        session.update_area_light(
            area_light,
            position=(2.5 * np.cos(angle), 5.0, -2.0 * np.sin(angle)),
            target=(0.0, 0.5, 0.0),
            size=(3.0, 3.0),
            color=(1.0, 0.9 + 0.1 * np.sin(angle), 0.85),
            strength=350.0 + 100.0 * np.sin(angle),
        )
        session.update_sun_light(
            sun_light,
            direction=(-0.4, -1.0, 0.25 + 0.15 * np.cos(angle)),
            color=(1.0, 0.95, 0.9),
            strength=1.5,
            angle=0.5,
        )

        seed = 1000 + frame
        start = time.perf_counter()
        session.render(seed=seed)
        elapsed = time.perf_counter() - start

        save_passes(session, args.output, frame, seed)
        if not args.no_preview:
            pr.save_png(
                os.path.join(args.output, f"frame_{frame:04d}.png"),
                session.beauty(),
            )

        print(f"frame {frame:04d}: {elapsed:.3f}s seed={seed}")

    session.close()
    print("wrote", args.output)


if __name__ == "__main__":
    main()
