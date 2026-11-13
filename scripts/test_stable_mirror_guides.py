"""A/B the experimental single mirror guide in the 1-SPP DLSS viewer.

Glass and rough reflectors are exclusion tests, not supported stable planes.
Requires OCIO to be set. Leaves fixtures, frame metrics and captures in output.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test_outputs/stable_mirror'
OUT.mkdir(parents=True, exist_ok=True)


def mesh(root, name, shader, points):
    state = ET.SubElement(root, 'state', shader=shader, interpolation='flat')
    obj = ET.SubElement(state, 'object', name=name)
    ET.SubElement(obj, 'mesh', P=' '.join(str(x) for p in points for x in p),
                  nverts='3 3', verts='0 1 2 0 2 3')


def fixture(kind):
    root = ET.Element('cycles')
    ET.SubElement(root, 'integrator', max_bounce='8', use_adaptive_sampling='false')
    camera = ET.SubElement(root, 'transform', translate='0 1.5 -4')
    ET.SubElement(camera, 'camera', width='512', height='512', type='perspective', fov='0.785398163')
    bg = ET.SubElement(root, 'background')
    ET.SubElement(bg, 'background', name='bg', color='1 1 1', strength='1')
    ET.SubElement(bg, 'connect', attrib={'from': 'bg background', 'to': 'output surface'})
    ET.SubElement(root, 'light', light_type='background', strength='1 1 1', use_mis='true')
    shader = ET.SubElement(root, 'shader', name='optic')
    tag = 'glass_bsdf' if kind == 'glass' else 'glossy_bsdf'
    attrs = dict(name='bsdf', color='1 1 1', roughness='0.25' if kind == 'rough' else '0', distribution='ggx')
    if kind == 'glass': attrs['IOR'] = '1.5'
    ET.SubElement(shader, tag, **attrs)
    ET.SubElement(shader, 'connect', attrib={'from': 'bsdf BSDF', 'to': 'output surface'})
    for i, color in enumerate(['0.7 0.03 0.03', '0.03 0.7 0.03', '0.03 0.03 0.7', '0.8 0.8 0.8']):
        shader = ET.SubElement(root, 'shader', name='color' + str(i))
        ET.SubElement(shader, 'diffuse_bsdf', name='bsdf', color=color, roughness='0')
        ET.SubElement(shader, 'connect', attrib={'from': 'bsdf BSDF', 'to': 'output surface'})
    mesh(root, 'optic_front', 'optic', [(-2, -.5, 2), (-2, 3.5, 2), (2, 3.5, 2), (2, -.5, 2)])
    if kind == 'glass':
        mesh(root, 'optic_back', 'optic', [(-2, -.5, 2.15), (2, -.5, 2.15), (2, 3.5, 2.15), (-2, 3.5, 2.15)])
    z = 7 if kind == 'glass' else -7
    for x in range(12):
        for y in range(8):
            a, b = (x - 6) * .7, (y - 2) * .7
            mesh(root, f'tile{x}_{y}', 'color' + str((x + y) % 4),
                 [(a, b, z), (a + .69, b, z), (a + .69, b + .69, z), (a, b + .69, z)])
    path = OUT / f'{kind}.xml'
    ET.ElementTree(root).write(path, encoding='unicode')
    return path


def run(scene, name, frames, stable=False, reference=False):
    command = [str(ROOT / 'install/photara_viewer.exe'), '--size', '512', '--spp', '1',
               '--smoke-frames', str(frames), '--metrics', str(OUT / f'{name}.csv'),
               '--screenshot', str(OUT / f'{name}.bmp')]
    if reference:
        command += ['--mode', 'REFERENCE', '--reference-samples', '256', '--camera-x', '-0.5196152423']
    else:
        command += ['--denoiser', 'DLSS', '--camera-sweep']
    if stable: command += ['--stable-mirror-guide']
    with (OUT / f'{name}.log').open('w') as stdout, (OUT / f'{name}.err').open('w') as stderr:
        subprocess.run(command + [str(scene)], cwd=ROOT, stdout=stdout, stderr=stderr,
                       check=True, timeout=180)
    warnings = (OUT / f'{name}.err').read_text().replace(
        'libpng warning: iCCP: known incorrect sRGB profile', '').strip()
    assert not warnings, (name, warnings)
    with (OUT / f'{name}.csv').open() as stream: rows = list(csv.DictReader(stream))
    assert len(rows) == frames
    if not reference:
        assert all(int(r['spp']) == 1 and int(r['dlss_frames']) == i + 1 for i, r in enumerate(rows))
        assert int(rows[-1]['dlss_resets']) == 1
        assert all(int(r['cpu_pixel_readback_bytes']) == 0 for r in rows)
        assert all(int(r['stable_mirror_guide']) == int(stable) for r in rows)
    return np.array(Image.open(OUT / f'{name}.bmp'), dtype=np.float32)[..., :3], rows


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cases', nargs='+', choices=('mirror', 'glass', 'rough'),
                    default=['mirror', 'glass', 'rough'])
args = parser.parse_args()
results = json.loads((OUT / 'summary.json').read_text()) if (OUT / 'summary.json').exists() else []
results = [r for r in results if r['kind'] not in args.cases]
for kind in args.cases:
    scene = fixture(kind)
    reference, _ = run(scene, f'{kind}_reference', 1, reference=True)
    images = []
    result = dict(kind=kind, runs=[])
    for stable in (False, True):
        image, rows = run(scene, f'{kind}_{"stable" if stable else "legacy"}', 181, stable=stable)
        times = [float(rows[i]['present_timestamp_ms']) - float(rows[i - 1]['present_timestamp_ms']) for i in range(61, 181)]
        result['runs'].append(dict(stable=stable, display_mse=float(np.mean((image-reference)**2)),
                                   motion_ms=statistics.mean(times), fps=1000/statistics.mean(times)))
        images.append(image)
    result['ab_mse'] = float(np.mean((images[0] - images[1])**2))
    if kind != 'mirror':
        assert result['ab_mse'] < .01, f'{kind}: excluded path changed its output'
    else:
        assert result['ab_mse'] > .01, 'Mirror prototype did not change reconstructed output'
    results.append(result)
    (OUT / 'summary.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(result), flush=True)
