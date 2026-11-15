"""Measure GPU temporal history, including Python result capture."""
import argparse
import json
from pathlib import Path
import sys
import time

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
    with plane_scene(history_samples=args.history) as scene:
        scene.resize(args.size, args.size * 3 // 4)
        timings = []
        for i in range(args.frames):
            scene.camera.location = (i*0.002, 0, -4)
            started = time.perf_counter()
            image = scene.render().beauty
            timings.append((time.perf_counter()-started)*1000)
            if not np.isfinite(image).all():
                raise RuntimeError("Non-finite GPU history output")
        stats = scene.history_stats()  # Explicit diagnostic read outside timed frames.
        report = {"GPU": dict(median_frame_ms=float(np.median(timings[8:])),
                              p95_frame_ms=float(np.percentile(timings[8:], 95)), **stats)}
    assert report["GPU"]["backend"] == "GPU"
    assert report["GPU"]["gpu_updates"] == args.frames
    assert report["GPU"]["reconstruction_pixel_readback_bytes"] == 0
    report["settings"] = vars(args) | {"output": str(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
