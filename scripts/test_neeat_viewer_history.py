"""Verify viewer NEE-AT history survives motion and resets on cuts/mode changes."""
import csv
import json
import os
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "test_outputs/neeat_viewer_history"
    output.mkdir(parents=True, exist_ok=True)
    startup = None
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
    results = {}
    for name, count, options in [
        ("motion_cut", 96, ["--camera-sweep", "--camera-cut-frame", "85"]),
        ("mode_resize", 32, ["--smoke-cycle"]),
    ]:
        metrics = output / f"{name}.csv"
        command = [str(root / "install/photara_viewer.exe"), "--size", "128",
                   "--spp", "1", "--light-sampler", "NEE_AT", "--smoke-frames", str(count),
                   "--metrics", str(metrics), "--screenshot", str(output / f"{name}.bmp"),
                   *options, str(root / "examples/scene_many_lights_neeat.xml")]
        result = subprocess.run(command, cwd=root, capture_output=True, timeout=120,
                                startupinfo=startup)
        log = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        (output / f"{name}.log").write_text(log, encoding="utf-8")
        assert result.returncode == 0, log
        assert "ERROR" not in log, log
        with metrics.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count, len(rows)
        assert all(int(r["cpu_pixel_readback_bytes"]) == 0 for r in rows)
        resets = [i for i, r in enumerate(rows) if int(r["neeat_reset"])]
        if name == "motion_cut":
            assert resets == [0, 85], resets
            moving = rows[60:85]
            assert len({r["camera_x"] for r in moving}) > 20
            assert all(int(r["neeat_reset"]) == 0 for r in moving)
            assert [int(r["sample_subset_offset"]) for r in rows] == list(range(count))
            # The film still resets while moving: retaining sampling history
            # must not smear images by averaging different camera positions.
            assert all(int(r["history_samples"]) == 1 for r in rows[61:85])
        else:
            assert resets == [0, 8, 9, 24, 25], resets
            assert rows[8]["mode"] == rows[24]["mode"] == "REFERENCE"
            assert rows[9]["mode"] == rows[25]["mode"] == "REALTIME"
            assert rows[16]["camera_x"] != rows[15]["camera_x"]
            assert int(rows[16]["neeat_reset"]) == 0
            assert int(rows[16]["sample_subset_offset"]) == int(rows[15]["sample_subset_offset"]) + 1
        results[name] = {"frames": len(rows), "reset_frames": resets,
                         "cpu_pixel_readback_bytes": 0}
    (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
