"""Compare explicit CPU reference with GPU history, including Python result capture."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "photara/python/tests"))
from test_history_samples import plane_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--output", type=Path, default=ROOT / "test_outputs/gpu_history_metrics.json")
    args = parser.parse_args()
    if args.frames < 10 or args.size < 32 or not 1 <= args.history <= 64:
        parser.error("requires frames >= 10, size >= 32, history in [1, 64]")
    report, images = {}, []
    for backend in ("GPU", "CPU"):
        env = dict(os.environ)
        env.pop("PHOTARA_HISTORY_CPU", None)
        if backend == "CPU": env["PHOTARA_HISTORY_CPU"] = "1"
        with patch.dict(os.environ, env, clear=True), plane_scene(history_samples=args.history) as scene:
            scene.resize(args.size, args.size * 3 // 4)
            timings = []
            for i in range(args.frames):
                scene.camera.location = (i*0.002, 0, -4)
                started = time.perf_counter()
                image = scene.render().beauty.copy()
                timings.append((time.perf_counter()-started)*1000)
            stats = scene.history_stats()  # Explicit diagnostic read outside timed frames.
            report[backend] = dict(median_frame_ms=float(np.median(timings[8:])),
                                   p95_frame_ms=float(np.percentile(timings[8:], 95)), **stats)
            images.append(image)
    np.testing.assert_allclose(images[0], images[1], rtol=3e-5, atol=3e-6)
    assert report["GPU"]["gpu_updates"] == args.frames
    assert report["GPU"]["reconstruction_pixel_readback_bytes"] == 0
    report["max_abs_difference"] = float(np.max(np.abs(images[0]-images[1])))
    report["settings"] = vars(args) | {"output": str(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
