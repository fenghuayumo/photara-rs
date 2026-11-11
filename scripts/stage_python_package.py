#!/usr/bin/env python3
"""Stage the built Photara Python module, DLLs and kernels into the package.

This consumes the release build (build/) and install/ tree and produces a
self-contained `photara_rs` package directory that can be turned into a
wheel with `pip wheel photara/python`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "photara" / "python" / "photara_rs"
INSTALL = REPO / "install"
DUMPBIN = Path(
    r"D:/ProgramTool/VS2022/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/dumpbin.exe"
)


def search_dirs() -> list[Path]:
    dirs = [INSTALL]
    lib_root = REPO / "lib" / "windows_x64"
    for sub in lib_root.iterdir() if lib_root.is_dir() else []:
        if sub.is_dir():
            for name in ("bin", "lib"):
                candidate = sub / name
                if candidate.is_dir():
                    dirs.append(candidate)
    dirs.append(lib_root)
    return dirs


DIRS = search_dirs()


def find_dll(name: str) -> Path | None:
    for directory in DIRS:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def dependencies(path: Path) -> list[str]:
    output = subprocess.check_output(
        [str(DUMPBIN), "/dependents", str(path)],
        text=True,
        errors="ignore",
    )
    result: list[str] = []
    reading = False
    for line in output.splitlines():
        if "Image has the following dependencies" in line:
            reading = True
            continue
        if reading:
            entry = line.strip()
            if not entry:
                continue
            if entry.startswith("Summary"):
                break
            result.append(entry)
    return result


def main() -> int:
    pyd = PACKAGE / ("_photara" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not pyd.is_file():
        print(f"No {pyd.name} in {PACKAGE}; build for this interpreter first.")
        return 1
    others = [p for p in PACKAGE.glob("_photara*.pyd") if p != pyd]
    if others:
        print("Foreign ABI extensions found; move them out before staging: "
              + ", ".join(p.name for p in others))
        return 1
    if any("_d.dll" in dep for dep in dependencies(pyd)):
        print(f"{pyd.name} links debug external libraries; build the Release configuration instead.")
        return 1

    bin_dir = PACKAGE / "bin"
    lib_dir = PACKAGE / "lib"
    bin_dir.mkdir(exist_ok=True)
    lib_dir.mkdir(exist_ok=True)
    for stale in list(bin_dir.glob("*.dll")):
        stale.unlink()

    queue = [pyd]
    seen: set[str] = set()
    copied: list[str] = []
    while queue:
        current = queue.pop()
        for dependency in dependencies(current):
            key = dependency.lower()
            if key in seen:
                continue
            seen.add(key)
            found = find_dll(dependency)
            if found is None:
                # System DLL, VC runtime or the Python interpreter DLL.
                continue
            target = bin_dir / dependency
            shutil.copy2(found, target)
            copied.append(dependency)
            queue.append(found)

    # OpenImageDenoise loads its devices dynamically, and Cycles can fall back
    # to OIDN if the OptiX denoiser is unavailable.
    for oidn in INSTALL.glob("OpenImageDenoise*.dll"):
        target = bin_dir / oidn.name
        if not target.exists():
            shutil.copy2(oidn, target)
            copied.append(oidn.name)

    blob_dir = INSTALL / "lib"
    blobs = sorted(blob_dir.glob("*.zst"))
    if not blobs:
        print(f"No kernel blobs in {blob_dir}; run cmake --install build first.")
        return 1
    for stale in list(lib_dir.glob("*.zst")):
        stale.unlink()
    for blob in blobs:
        shutil.copy2(blob, lib_dir / blob.name)

    size = sum(p.stat().st_size for p in bin_dir.glob("*.dll"))
    size += sum(p.stat().st_size for p in lib_dir.glob("*.zst"))
    print(f"pyd: {pyd.name}")
    print(f"copied {len(copied)} DLLs, {len(blobs)} kernel blobs")
    print(f"runtime payload: {size / (1024 ** 2):.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
