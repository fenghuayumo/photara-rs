"""Apply the narrow Cycles display-driver bridge to a compatible Blender checkout."""

import argparse
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ABI = '4d9862575ebf7cca74c48c4b10099fcc01fc9c58b1e82278035b74cbb6aac170'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender-source', type=Path, required=True)
    args = parser.parse_args()
    source = args.blender_source.resolve()
    for header in (ROOT / 'src/session/display_driver.h',
                   source / 'intern/cycles/session/display_driver.h'):
        digest = hashlib.sha256(header.read_bytes()).hexdigest()
        if digest != EXPECTED_ABI:
            parser.error(f'DisplayDriver ABI differs: {header}. Port and verify the bridge first.')
    patch = ROOT / 'scripts/blender_photara_display_bridge.patch'
    base = ['git', '-C', str(source), 'apply']
    if subprocess.run(base + ['--reverse', '--check', str(patch)], capture_output=True).returncode == 0:
        print('Photara display bridge is already applied.')
        return
    subprocess.run(base + ['--check', str(patch)], check=True)
    subprocess.run(base + [str(patch)], check=True)
    print('Applied display bridge. Rebuild Blender before installing the Photara wheel/add-on.')


if __name__ == '__main__':
    main()
