"""Cornell Box baseline: shared geometry for XML and Python, warm frame timings.

Run with a Python matching the built photara_rs extension and PYTHONPATH=photara/python.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np


COLORS = {"white": (.73, .73, .73), "red": (.65, .05, .05),
          "green": (.12, .45, .15), "light": (1., 1., 1.)}


def geometry():
    faces = [
        ("floor", "white", [(-1, 0, 0), (-1, 0, 2), (1, 0, 2), (1, 0, 0)]),
        ("ceiling", "white", [(-1, 2, 0), (1, 2, 0), (1, 2, 2), (-1, 2, 2)]),
        ("back", "white", [(-1, 0, 2), (-1, 2, 2), (1, 2, 2), (1, 0, 2)]),
        ("left", "red", [(-1, 0, 0), (-1, 2, 0), (-1, 2, 2), (-1, 0, 2)]),
        ("right", "green", [(1, 0, 0), (1, 0, 2), (1, 2, 2), (1, 2, 0)]),
        ("emitter", "light", [(-.3, 1.99, .7), (.3, 1.99, .7),
                                (.3, 1.99, 1.3), (-.3, 1.99, 1.3)]),
    ]
    for name, cx, cz, height, angle in [("short", -.4, .65, .6, -18),
                                       ("tall", .4, 1.35, 1.2, 17)]:
        v = np.array([[-.28, 0, -.28], [.28, 0, -.28], [.28, 0, .28],
                      [-.28, 0, .28], [-.28, height, -.28], [.28, height, -.28],
                      [.28, height, .28], [-.28, height, .28]])
        a = np.deg2rad(angle)
        v = v @ np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0],
                          [np.sin(a), 0, np.cos(a)]])
        v += (cx, 0, cz)
        for i, indices in enumerate([(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6),
                                     (3, 0, 4, 7), (4, 5, 6, 7), (3, 2, 1, 0)]):
            faces.append((f"{name}_{i}", "white", v[list(reversed(indices))].tolist()))
    return faces


def write_xml(path):
    root = ET.Element("cycles")
    ET.SubElement(root, "integrator", max_bounce="8", use_adaptive_sampling="false")
    transform = ET.SubElement(root, "transform", translate="0 1 -3.2")
    ET.SubElement(transform, "camera", width="512", height="512", type="perspective",
                  fov=str(np.deg2rad(39)))
    bg = ET.SubElement(root, "background")
    ET.SubElement(bg, "background", name="bg", color="0 0 0", strength="0")
    ET.SubElement(bg, "connect", attrib={"from": "bg background", "to": "output surface"})
    for name, color in COLORS.items():
        shader = ET.SubElement(root, "shader", name=name)
        if name == "light":
            ET.SubElement(shader, "emission", name="bsdf", color="1 1 1", strength="15")
            socket = "emission"
        else:
            ET.SubElement(shader, "principled_bsdf", name="bsdf",
                          base_color=" ".join(map(str, color)), roughness="0.7")
            socket = "bsdf"
        ET.SubElement(shader, "connect", attrib={"from": f"bsdf {socket}",
                                                 "to": "output surface"})
    for name, material, vertices in geometry():
        obj = ET.SubElement(root, "object", name=name)
        state = ET.SubElement(obj, "state", shader=material)
        ET.SubElement(state, "mesh", P=" ".join(str(x) for v in vertices for x in v),
                      nverts="3 3", verts="0 1 2 0 2 3")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")


def build_scene(size, samples, denoiser):
    import photara_rs as pr
    scene = pr.Scene(size, size, samples=samples, max_bounces=8,
                     adaptive_sampling=False, denoise=denoiser != "NONE",
                     denoiser="AUTO" if denoiser == "NONE" else denoiser)
    scene.camera.location = (0, 1, -3.2)
    scene.camera.target = (0, 1, 1)
    scene.camera.fov = 39
    scene.world.color = (0, 0, 0)
    materials = {}
    for name, color in COLORS.items():
        materials[name] = scene.materials.new(name, base_color=color, roughness=.7,
                                              emission_color=(1, 1, 1),
                                              emission_strength=15 if name == "light" else 0)
    for i, (name, material, vertices) in enumerate(geometry(), 1):
        mesh = scene.meshes.new(name, vertices, [(0, 1, 2), (0, 2, 3)],
                                material=materials[material])
        scene.objects.new(name, mesh, object_id=i)
    return scene


def save_preview(path, linear):
    from PIL import Image
    # Fixed exposure and simple display transform shared by all comparisons.
    rgb = np.maximum(linear[..., :3], 0)
    rgb = rgb / (1 + rgb)
    rgb = np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - .055)
    Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--reference-samples", type=int, default=256)
    parser.add_argument("--samples-per-frame", type=int, default=1)
    parser.add_argument("--denoiser", choices=["NONE", "AUTO", "DLSS"], default="NONE")
    parser.add_argument("--output", type=Path, default=Path("test_outputs/cornell_realtime"))
    parser.add_argument("--xml-only", action="store_true")
    args = parser.parse_args()
    if min(args.size, args.frames, args.reference_samples) <= 0:
        parser.error("size, frames and reference-samples must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    write_xml(args.output / "cornell_box.xml")
    if args.xml_only:
        return
    with build_scene(args.size, args.reference_samples, args.denoiser) as scene:
        start = time.perf_counter()
        reference = scene.render().beauty.copy()
        reference_ms = (time.perf_counter() - start) * 1000
        scene.configure_render_mode("REALTIME", samples_per_frame=args.samples_per_frame)
        scene.render()  # Warm allocations, kernels and reconstruction.
        times, images = [], []
        for _ in range(args.frames):
            start = time.perf_counter()
            image = scene.render().beauty.copy()
            times.append((time.perf_counter() - start) * 1000)
            if not np.isfinite(image).all():
                raise RuntimeError("RealTime produced non-finite pixels")
            images.append(image)
        average = np.mean(images, axis=0)
        save_preview(args.output / "reference.png", reference)
        save_preview(args.output / "realtime_last.png", images[-1])
        save_preview(args.output / "realtime_average.png", average)
        np.savez_compressed(args.output / "linear.npz", reference=reference,
                            realtime_last=images[-1], realtime_average=average)
        result = {"size": args.size, "denoiser": args.denoiser,
                  "samples_per_frame": args.samples_per_frame, "frames": args.frames,
                  "reference_samples": args.reference_samples, "reference_cold_ms": reference_ms,
                  "realtime_warm_median_ms": float(np.median(times)),
                  "realtime_warm_p95_ms": float(np.percentile(times, 95)),
                  "realtime_frame_ms": times,
                  "last_rgb_rmse": float(np.sqrt(np.mean((images[-1][..., :3] - reference[..., :3]) ** 2))),
                  "average_rgb_rmse": float(np.sqrt(np.mean((average[..., :3] - reference[..., :3]) ** 2))),
                  "timing_scope": "Scene.render including CPU pass capture; excludes PNG/NPZ writes",
                  "average_scope": "Offline arithmetic mean, not renderer temporal reconstruction",
                  "algorithms": "OptiX frame baseline + Light Tree; no NEE-AT/ReSTIR/cache yet"}
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
