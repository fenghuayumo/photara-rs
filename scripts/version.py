#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read, check or bump the shared Photara MAJOR.MINOR.PATCH version.

python scripts/version.py --check
python scripts/version.py 0.3.0
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)'


def validate_version(value: str) -> str:
    if not re.fullmatch(VERSION_PATTERN, value):
        raise ValueError('Version must be MAJOR.MINOR.PATCH, without a v prefix or leading zeroes')
    return value


def read_version(root: Path = ROOT) -> str:
    return validate_version((root / 'VERSION').read_text(encoding='utf-8').strip())


def addon_source_version(root: Path = ROOT) -> tuple[int, int, int]:
    tree = ast.parse((root / 'photara_rs_engine/__init__.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'bl_info' for target in node.targets
        ):
            return ast.literal_eval(node.value)['version']
    raise ValueError('Blender bl_info metadata is missing')


def check_version(root: Path = ROOT) -> str:
    version = read_version(root)
    if addon_source_version(root) != tuple(map(int, version.split('.'))):
        raise ValueError('Blender metadata differs from VERSION; run scripts/version.py --sync')
    return version


def sync_addon(version: str, root: Path = ROOT) -> None:
    path = root / 'photara_rs_engine/__init__.py'
    source = path.read_text(encoding='utf-8')
    # Blender discovers add-ons by parsing this literal without importing bpy.
    value = ', '.join(version.split('.'))
    source, count = re.subn(r'("version"\s*:\s*)\(\d+,\s*\d+,\s*\d+\)',
                            lambda match: match[1] + f'({value})', source, count=1)
    if count != 1:
        raise ValueError('Could not update Blender version metadata')
    if source != path.read_text(encoding='utf-8'):
        path.write_text(source, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version', nargs='?')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--check', action='store_true')
    actions.add_argument('--sync', action='store_true')
    args = parser.parse_args()
    if args.version and (args.check or args.sync):
        parser.error('Choose a new version, --check, or --sync')
    try:
        if args.version:
            version = validate_version(args.version)
            sync_addon(version)
            (ROOT / 'VERSION').write_text(version + '\n', encoding='utf-8')
        elif args.sync:
            sync_addon(read_version())
        print(check_version())
    except (OSError, ValueError, SyntaxError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
