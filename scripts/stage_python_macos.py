#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage installed Metal runtime sources for the macOS Python package.

Build with PHOTARA_BUILD_PYTHON=ON, then install the CMake build first.
Requires delocate to bundle external dylibs and dynamically loaded OIDN devices.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-dir', type=Path)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        parser.error('This helper requires macOS')
    try:
        from delocate.delocating import delocate_path
    except ImportError:
        parser.error('Install delocate in this interpreter: python -m pip install delocate')
    repo = Path(__file__).resolve().parent.parent
    install = args.install_dir or repo / 'install'
    package = repo / 'photara/python/photara_rs'
    module = package / ('_photara' + sysconfig.get_config_var('EXT_SUFFIX'))
    if not module.is_file():
        parser.error(f'Build the extension for this interpreter first: {module}')
    source = install / 'source'
    if not (source / 'kernel/device/metal/kernel.metal').is_file():
        parser.error(f'Metal runtime sources missing from {source}; install the build first')
    shutil.copytree(source, package / 'source', dirs_exist_ok=True)
    # Blender precompiled dependencies may include dynamically loaded OIDN devices.
    for library in (install / 'lib').glob('*.dylib*'):
        target = package / 'lib' / library.name
        target.parent.mkdir(exist_ok=True)
        target.unlink(missing_ok=True)
        shutil.copy2(library, target)
    # OIDN loads device libraries at runtime; dependency scanners cannot discover
    # them from the extension's load commands. Stage them beside the libraries
    # delocate will bundle with --lib-sdir lib.
    links = subprocess.check_output(['otool', '-L', str(module)], text=True)
    for line in links.splitlines()[1:]:
        dependency = Path(line.strip().split(' (', 1)[0])
        if dependency.is_absolute() and dependency.name.startswith('libOpenImageDenoise.'):
            for library in dependency.parent.glob('libOpenImageDenoise_device_*.dylib'):
                target = package / 'lib' / library.name
                target.parent.mkdir(exist_ok=True)
                target.unlink(missing_ok=True)
                shutil.copy2(library, target)
    delocate_path(str(package), str(package / 'lib'),
                  lib_filt_func='dylibs-only', sanitize_rpaths=True)
    print(f'Staged Metal sources in {package / "source"}')
    print('After building the wheel, run delocate-wheel --lib-sdir lib -w dist/repaired dist/<wheel>.whl')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
