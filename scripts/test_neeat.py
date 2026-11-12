"""Compare experimental temporal NEE-AT against Light Tree at equal frame budgets.

The scene mixes analytic lights with an emissive mesh and an occluder. A single
visible key light competes with many occluded lights, making feedback useful.
Run with a Python matching photara_rs; outputs are linear data and metrics.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import photara_rs as pr


def build_scene(size, samples, lights=32):
    scene = pr.Scene(size, size, samples=samples, max_bounces=4,
                     denoise=False, adaptive_sampling=False)
    scene.world.color = (0, 0, 0)
    scene.camera.location = (0, 2.4, -4)
    scene.camera.target = (0, .3, 1)
    scene.camera.fov = 48
    white = scene.materials.new("white", base_color=(.7, .7, .7), roughness=.8)
    glow = scene.materials.new("glow", base_color=(0, 0, 0),
                               emission_color=(.2, .5, 1), emission_strength=3)

    def quad(name, positions, material):
        mesh = scene.meshes.new(name, positions, [(0, 1, 2), (0, 2, 3)], material=material)
        scene.objects.new(name, mesh)

    quad("floor", [(-3, 0, -1), (-3, 0, 5), (3, 0, 5), (3, 0, -1)], white)
    quad("ceiling_blocker", [(-50, 3, -50), (50, 3, -50), (50, 3, 50), (-50, 3, 50)], white)
    quad("back", [(-3, 0, 4), (-3, 3, 4), (3, 3, 4), (3, 0, 4)], white)
    quad("occluder", [(.2, 0, 1), (.2, 1.4, 1), (1.1, 1.4, 1), (1.1, 0, 1)], white)
    quad("emissive", [(-1.4, .5, 3.9), (-1.4, 1.3, 3.9),
                       (-.6, 1.3, 3.9), (-.6, .5, 3.9)], glow)
    scene.lights.new_area("key", position=(-.7, 2, .5), target=(0, 0, 1),
                          size=(.5, .5), strength=180)
    for i in range(lights - 1):
        angle = 2 * np.pi * i / max(1, lights - 1)
        scene.lights.new_area(f"far_{i}", position=(12*np.cos(angle), 6, 12*np.sin(angle)+1),
                              target=(0, 0, 1), size=(.5, .5), strength=180)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--reference-samples", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path("test_outputs/neeat"))
    args = parser.parse_args()
    if min(args.size, args.frames, args.reference_samples) <= 0 or args.warmup < 0:
        parser.error("positive size/frames/reference samples required; warmup must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, arrays = {}, {}
    with build_scene(args.size, args.reference_samples) as scene:
        reference = scene.render().rgb.copy()
        arrays["reference"] = reference
        for name, enabled in [("light_tree", False), ("nee_at", True)]:
            scene.configure_neeat(enabled)
            scene.configure_render_mode("REALTIME", samples_per_frame=1)
            for _ in range(args.warmup):
                scene.render()
            images, timings = [], []
            for _ in range(args.frames):
                start = time.perf_counter()
                image = scene.render().rgb.copy()
                timings.append((time.perf_counter()-start)*1000)
                if not np.isfinite(image).all():
                    raise RuntimeError("Non-finite output")
                images.append(image)
            images = np.asarray(images)
            mean = images.mean(axis=0)
            arrays[name] = mean
            metrics[name] = {
                "stats": scene.neeat_stats(),
                "warm_median_ms": float(np.median(timings)),
                "mean_single_frame_mse": float(np.mean((images-reference)**2)),
                "temporal_mean_rmse": float(np.sqrt(np.mean((mean-reference)**2))),
                "mean_luminance": float(mean.mean()),
            }
        metrics["reference_luminance"] = float(reference.mean())
        metrics["settings"] = vars(args) | {"output": str(args.output)}
        metrics["timing_scope"] = "Scene.render including CPU AOV capture, excluding file writes"
        metrics["algorithm"] = "Experimental global + screen-tile analytic-light feedback + tree mixture"
        np.savez_compressed(args.output / "linear.npz", **arrays)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
