#!/usr/bin/env python3
"""Install a matching wheel and enable Photara in an existing Blender build.

Run with an ordinary Python, not Blender's --python-expr. The native wheel is
installed while Blender is closed, then a fresh Blender process enables the
add-on and saves preferences without changing startup scenes or other add-ons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def enable_addon(archive):
    import bpy
    import addon_utils
    import importlib

    name = "photara_rs_engine"
    addon_utils.disable(name, default_set=True)
    for module in list(sys.modules):
        if module == name or module.startswith(name + "."):
            del sys.modules[module]
    bpy.ops.preferences.addon_install(filepath=str(archive), overwrite=True)
    importlib.invalidate_caches()
    addon_utils.enable(name, default_set=True, persistent=True)
    if name not in bpy.context.preferences.addons:
        raise RuntimeError("Photara add-on did not enable")
    import photara_rs_engine

    error = photara_rs_engine.ensure_photara_module()
    if error:
        raise RuntimeError(error)
    import photara_rs

    with photara_rs.Scene(16, 16, samples=1, device="OPTIX") as scene:
        print("PHOTARA_DEVICE:", scene.device_name)
    bpy.ops.wm.save_userpref()
    print("PHOTARA_INSTALL_OK")


def main():
    if "--" in sys.argv and "--enable-addon" in sys.argv:
        enable_addon(Path(sys.argv[-1]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True,
                        help="Path to blender.exe or its bin directory")
    parser.add_argument("--wheel", type=Path,
                        help="Optional wheel override (otherwise auto-selected for Blender's Python)")
    parser.add_argument("--addon", type=Path,
                        help="Optional add-on zip override")
    args = parser.parse_args()
    blender = args.blender.resolve()
    if blender.is_dir():
        blender /= "blender.exe"
    if not blender.is_file():
        parser.error(f"Blender executable does not exist: {blender}")
    probe = subprocess.run([
        str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
        "--python-expr", "import json,sys; print('PHOTARA_PYTHON:'+json.dumps("
        "{'executable':sys.executable,'tag':f'cp{sys.version_info.major}{sys.version_info.minor}'}))",
    ], check=True, text=True, capture_output=True)
    info = next(json.loads(line.split(":", 1)[1]) for line in probe.stdout.splitlines()
                if line.startswith("PHOTARA_PYTHON:"))
    tag = info["tag"]
    wheels = sorted((ROOT / "dist").glob(f"photara_rs-*-{tag}-{tag}-win_amd64.whl"))
    wheel = args.wheel.resolve() if args.wheel else (wheels[-1] if wheels else None)
    if wheel is None or not wheel.is_file():
        parser.error(f"Build a {tag} Windows x64 wheel in dist/ first")
    if not wheel.name.endswith(f"-{tag}-{tag}-win_amd64.whl"):
        parser.error(f"Blender requires a {tag} Windows x64 wheel: {wheel.name}")
    if args.addon:
        archive = args.addon.resolve()
    else:
        subprocess.run([sys.executable, str(ROOT / "scripts/package_blender_addon.py")], check=True)
        from package_blender_addon import addon_version

        archive = ROOT / "dist" / f"photara_blender_addon-{addon_version()}.zip"
    if not archive.is_file():
        parser.error(f"Add-on archive does not exist: {archive}")
    python = info["executable"]
    # Blender bundles NumPy. Preserve that version rather than replacing it.
    subprocess.run([python, "-m", "ensurepip"], check=True)
    subprocess.run([python, "-m", "pip", "install", "--upgrade", "--force-reinstall",
                    "--no-deps", str(wheel)], check=True)
    subprocess.run([python, "-c", "import numpy,photara_rs; print(photara_rs.__version__)"], check=True)
    subprocess.run([str(blender), "--background", "--python-exit-code", "1",
                    "--python", str(Path(__file__).resolve()), "--", "--enable-addon",
                    str(archive)], check=True)


if __name__ == "__main__":
    main()
