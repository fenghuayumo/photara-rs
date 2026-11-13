"""Check static DLSS stability and camera-motion history on the Sponza fixture.

Snapshots from deterministic runs ending at consecutive frame counts sample a
warmed sequence without adding readback stalls between its rendered frames.
Capture runs are quality diagnostics, not performance measurements.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=root / 'test_outputs/sponza_viewer/sponza.xml')
    parser.add_argument('--output', type=Path, default=root / 'test_outputs/dlss_temporal')
    parser.add_argument('--ocio', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.ocio:
        env['OCIO'] = str(args.ocio.resolve())
    startup = None
    if os.name == 'nt':
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE

    def run(name, count, *options):
        image, metrics = output / f'{name}.bmp', output / f'{name}.csv'
        result = subprocess.run([
            str(root / 'install/photara_viewer.exe'), '--size', '512', '--spp', '1',
            '--denoiser', 'DLSS', '--smoke-frames', str(count), '--metrics', str(metrics),
            '--screenshot', str(image), *options, str(args.scene.resolve()),
        ], cwd=root, env=env, startupinfo=startup, capture_output=True, timeout=120)
        log = (result.stdout + result.stderr).decode('utf-8', errors='replace')
        (output / f'{name}.log').write_text(log, encoding='utf-8')
        assert result.returncode == 0, log
        with metrics.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count
        assert int(rows[-1]['dlss_frames']) == count
        assert all(int(row['cpu_pixel_readback_bytes']) == 0 for row in rows)
        return np.asarray(Image.open(image).convert('RGB'), dtype=np.float32), rows

    images = []
    for count in range(120, 126):
        image, rows = run(f'static_{count}', count)
        assert int(rows[-1]['dlss_resets']) == 1
        images.append(image)
    differences = np.diff(np.asarray(images), axis=0)
    result = dict(static_temporal_mae=float(np.abs(differences).mean()),
                  static_temporal_rms=float(np.sqrt(np.mean(differences ** 2))))
    # The broken previous projection produced about 2.45/255 MAE here even
    # though reset/reuse counters looked correct. Check actual pixels as well.
    assert result['static_temporal_mae'] < .5, result
    assert result['static_temporal_rms'] < 2, result
    _, rows = run('motion_cut', 180, '--camera-sweep', '--camera-cut-frame', '150')
    assert int(rows[-1]['dlss_resets']) == 2
    assert [i for i, row in enumerate(rows) if int(row['temporal_reset'])] == [0, 150]
    # The sine sweep visits mirrored positions on each side of its peak.
    assert len({row['camera_x'] for row in rows[61:150]}) > 35
    result['motion_frames'] = len(rows)
    result['motion_resets'] = int(rows[-1]['dlss_resets'])
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
