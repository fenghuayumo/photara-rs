"""Finite 1-SPP native-resolution viewer runs, history checks and motion captures.

Requires the Sponza XML exported by export_sponza_xml.py and OCIO environment.
Measures successive application Present submissions, not monitor scanout.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=root / 'test_outputs/sponza_viewer/sponza.xml')
    parser.add_argument('--output', type=Path, default=root / 'test_outputs/sponza_dlss')
    parser.add_argument('--ocio', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.ocio:
        env['OCIO'] = str(args.ocio.resolve())
    results = []

    def run(name, size, count, *options):
        command = [str(root / 'install/photara_viewer.exe'), '--size', str(size),
                   '--spp', '1', '--smoke-frames', str(count), '--metrics', str(output / f'{name}.csv'),
                   '--screenshot', str(output / f'{name}.bmp'), *options, str(args.scene.resolve())]
        with (output / f'{name}.stdout.log').open('w') as stdout, (output / f'{name}.stderr.log').open('w') as stderr:
            subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr,
                           timeout=120, check=True)
        log = (output / f'{name}.stderr.log').read_text()
        if 'ERROR' in log or 'WARNING' in log:
            raise RuntimeError(f'{name}: inspect stderr log')
        with (output / f'{name}.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count, (name, len(rows))
        assert all(int(r['rendered_frames_presented']) == i + 1 for i, r in enumerate(rows))
        assert all(int(r['cpu_pixel_readback_bytes']) == 0 for r in rows)
        if '--denoiser' in options and options[options.index('--denoiser') + 1] == 'DLSS':
            assert all(int(r['spp']) == 1 and int(r['dlss_frames']) == i + 1 for i, r in enumerate(rows))
            assert int(rows[0]['dlss_resets']) == 1
            for i in range(1, len(rows)):
                expected_reset = int(rows[i]['temporal_reset'])
                assert int(rows[i]['dlss_resets']) - int(rows[i - 1]['dlss_resets']) == expected_reset
            assert all(int(r['history_samples']) == 1 for r in rows), 'DLSS output was averaged again'
        return rows

    for size in (720, 1080):
        rows = run(f'{size}_motion', size, 360, '--denoiser', 'DLSS',
                   '--camera-sweep', '--camera-cut-frame', '280')
        phases = {}
        for phase, begin, end in [('static', 20, 60), ('motion', 61, 240), ('settle', 241, 280), ('after_cut', 281, 360)]:
            intervals = [float(rows[i]['present_timestamp_ms']) - float(rows[i - 1]['present_timestamp_ms']) for i in range(begin, end)]
            phases[phase] = dict(mean_ms=statistics.mean(intervals), fps=1000 / statistics.mean(intervals),
                                 p95_ms=sorted(intervals)[int(.95 * (len(intervals) - 1))])
        result = dict(size=size, spp=1, denoiser='DLSS_RR', upscale_factor=1,
                      frames=len(rows), phases=phases, final=rows[-1])
        results.append(result)
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)

    for name, count in [('cold', 1), ('warm', 60), ('moving', 181), ('cut', 281)]:
        run(f'720_{name}', 720, count, '--denoiser', 'DLSS', '--camera-sweep', '--camera-cut-frame', '280')
    run('720_reference_origin', 720, 1, '--mode', 'REFERENCE', '--reference-samples', '256')
    run('720_reference_motion', 720, 1, '--mode', 'REFERENCE', '--reference-samples', '256',
        '--camera-x', '-0.5196152423')
    run('720_reference_cut', 720, 1, '--mode', 'REFERENCE', '--reference-samples', '256', '--camera-x', '1')
    print('History/presentation checks passed; captures saved.', flush=True)


if __name__ == '__main__':
    main()
