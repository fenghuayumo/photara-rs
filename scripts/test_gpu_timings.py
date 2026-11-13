"""Verify stage timestamps, disabled-stage reporting and timing-off image parity."""
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from summarize_gpu_timings import summarize


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'test_outputs/gpu_timings'
    out.mkdir(parents=True, exist_ok=True)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    results = {}
    images = {}
    for name, timing, denoiser, sampler, cpu in [
        ('off', False, 'NONE', 'NEE_AT', False),
        ('on', True, 'NONE', 'NEE_AT', False),
        ('dlss', True, 'DLSS', 'NEE_AT', False),
        ('tree', True, 'NONE', 'LIGHT_TREE', False),
        ('cpu_neeat', True, 'NONE', 'NEE_AT', True),
        ('cycle', True, 'NONE', 'NEE_AT', False),
    ]:
        env = os.environ.copy()
        env.pop('PHOTARA_NEEAT_CPU', None)
        if cpu:
            env['PHOTARA_NEEAT_CPU'] = '1'
        command = [str(root / 'install/photara_viewer.exe'), '--size', '128', '--spp', '1',
                   '--denoiser', denoiser, '--light-sampler', sampler, '--smoke-frames', '32',
                   '--metrics', str(out / f'{name}.csv'), '--screenshot', str(out / f'{name}.bmp')]
        if timing:
            command.append('--gpu-timings')
        if name == 'cycle':
            command.append('--smoke-cycle')
        command.append(str(root / 'examples/scene_many_lights_neeat.xml'))
        p = subprocess.run(command, cwd=root, env=env, startupinfo=startup,
                           capture_output=True, timeout=120)
        log = (p.stdout + p.stderr).decode('utf-8', errors='replace')
        (out / f'{name}.log').write_text(log, encoding='utf-8')
        assert p.returncode == 0, log
        with (out / f'{name}.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 32
        warm = rows[8:]
        for row in warm:
            assert int(row['gpu_timing_enabled']) == int(timing)
            assert int(row['cpu_pixel_readback_bytes']) == 0
            for stage in ('path_trace_gpu_ms', 'film_convert_gpu_ms', 'scene_update_host_ms'):
                if timing:
                    assert 0 < float(row[stage]) < float(row['observed_ms']) + 2, (stage, row)
                else:
                    assert row[stage] == '', row
            if timing:
                assert int(row['display_gpu_frame']) < int(row['frame'])
                assert 0 <= float(row['display_gpu_ms']) < 100
            else:
                assert row['display_gpu_ms'] == row['display_gpu_frame'] == ''
            for stage in ('gpu_queue_wait_host_ms', 'display_reader_wait_host_ms'):
                if timing:
                    assert 0 <= float(row[stage]) < float(row['observed_ms']) + 2
                else:
                    assert row[stage] == ''
            if timing and sampler == 'NEE_AT' and not cpu and row['mode'] == 'REALTIME':
                assert float(row['neeat_gpu_ms']) > 0
            else:
                assert row['neeat_gpu_ms'] == ''
            if timing and denoiser == 'DLSS':
                assert float(row['dlss_evaluate_gpu_ms']) > 0
                assert float(row['denoise_total_gpu_ms']) >= float(row['dlss_evaluate_gpu_ms'])
            else:
                assert row['denoise_total_gpu_ms'] == row['dlss_evaluate_gpu_ms'] == ''
        results[name] = summarize(rows, 8)
        images[name] = np.asarray(Image.open(out / f'{name}.bmp')).astype(np.int16)
    assert np.max(np.abs(images['off'] - images['on'])) <= 1, 'Timing changed rendered output'
    (out / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('GPU stages, delayed Vulkan attribution, disabled/fallback stages and image parity passed.')


if __name__ == '__main__':
    main()
