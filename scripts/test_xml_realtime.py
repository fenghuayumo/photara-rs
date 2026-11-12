"""Run original XML fixtures through Reference and both RealTime light samplers.

Uses installed executables and preserves original XML/include/resource paths.
Viewer timings include frame submission through GPU display update, not scanout.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess


def run(command, log, cwd, timeout):
    startup = None
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, timeout=timeout,
                                startupinfo=startup)
        text = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        log.write_text(text, encoding="utf-8")
        return result.returncode, text
    except subprocess.TimeoutExpired as error:
        text = ((error.stdout or b"") + (error.stderr or b"")).decode("utf-8", errors="replace")
        log.write_text(text + "\nProcess timed out", encoding="utf-8")
        return -1, text + "\nProcess timed out"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+", type=Path)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--reference-samples", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("test_outputs/xml_realtime"))
    args = parser.parse_args()
    if args.size < 16 or args.frames < 2 or args.reference_samples < 1 or args.timeout < 1:
        parser.error("size >= 16, frames >= 2, positive reference samples/timeout required")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    renderer, viewer = root / "install/photara_rs.exe", root / "install/photara_viewer.exe"
    if not renderer.is_file() or not viewer.is_file():
        parser.error("Build/install photara_rs.exe and photara_viewer.exe first")
    results = []
    for index, source in enumerate(args.scenes):
        source = source.resolve()
        folder = output / f"{index}_{source.stem}"
        folder.mkdir(parents=True, exist_ok=True)
        result = {"source": str(source), "artifacts": str(folder), "status": "passed"}
        results.append(result)
        if not source.is_file():
            result.update(status="failed", reason="Scene file is missing")
            continue
        code, text = run([str(renderer), "--background", "--quiet", "--device", "OPTIX",
                          "--width", str(args.size), "--height", str(args.size),
                          "--samples", str(args.reference_samples), "--output",
                          str(folder / "reference.png"), str(source)], folder / "reference.log",
                         root, args.timeout)
        # The legacy XML reader can log shader errors but still exit zero. Such
        # output must not be counted as a successful material/scene test.
        if code or "ERROR:" in text or not (folder / "reference.png").is_file():
            unsupported = 'Unknown shader node "osl_shader"' in text or "OSL node without" in text
            result.update(status="unsupported" if unsupported else "failed",
                          reason="OSL shader unavailable in this build" if unsupported else
                                 "Reference render failed; inspect reference.log")
            continue
        result["samplers"] = {}
        for sampler in ("LIGHT_TREE", "NEE_AT"):
            prefix = sampler.lower()
            metrics = folder / f"{prefix}.csv"
            image = folder / f"{prefix}.bmp"
            code, text = run([str(viewer), "--size", str(args.size), "--light-sampler", sampler,
                              "--smoke-frames", str(args.frames), "--metrics", str(metrics),
                              "--screenshot", str(image), str(source)], folder / f"{prefix}.log",
                             root, args.timeout)
            if code or "ERROR:" in text or not metrics.is_file() or not image.is_file():
                result.update(status="failed", reason=f"{sampler} viewer failed; inspect log")
                break
            with metrics.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != args.frames or [int(r["rendered_frames_presented"]) for r in rows] != list(range(1, args.frames+1)):
                result.update(status="failed", reason=f"{sampler} frame/presentation mismatch")
                break
            warm = rows[min(8, args.frames-1):]
            result["samplers"][sampler] = {
                "frames": len(rows), "presented": int(rows[-1]["rendered_frames_presented"]),
                "warm_worker_median_ms": statistics.median(float(r["worker_to_gpu_display_ms"]) for r in warm),
                "warm_observed_median_ms": statistics.median(float(r["observed_ms"]) for r in warm),
                "cpu_pixel_readback_bytes": sum(int(r["cpu_pixel_readback_bytes"]) for r in rows),
            }
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {"settings": {"size": args.size, "frames": args.frames,
                             "reference_samples": args.reference_samples}, "results": results,
               "notes": "Reference PNG and viewer BMP use different display transforms; no cross-image quality metric implied."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return int(any(result["status"] == "failed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
