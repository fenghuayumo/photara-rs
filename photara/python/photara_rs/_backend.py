"""Native module loading and resource discovery."""

from __future__ import annotations

import os
from typing import Optional

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PACKAGE_DIR, "..", "..", ".."))
_DEV_INSTALL_DIR = os.path.join(_REPO_ROOT, "install")
_IS_DEV_TREE = os.path.isfile(os.path.join(_REPO_ROOT, "photara", "CMakeLists.txt"))

_DLL_HANDLES: list[object] = []


def _add_dll_directory(path: Optional[str]) -> None:
    if not path or not os.path.isdir(path):
        return
    if hasattr(os, "add_dll_directory"):
        try:
            handle = os.add_dll_directory(path)
        except OSError:
            return
        _DLL_HANDLES.append(handle)


for _candidate in (
    os.environ.get("PHOTARA_DLL_DIR"),
    os.path.join(_PACKAGE_DIR, "bin"),
    _DEV_INSTALL_DIR if _IS_DEV_TREE else None,
):
    _add_dll_directory(_candidate)

try:
    from . import _photara as _core
except ImportError as exc:  # pragma: no cover - depends on the built artifact.
    raise ImportError(
        "Could not load the photara_rs native module. The package must contain "
        "a matching _photara*.pyd under photara_rs/, and the runtime DLLs must "
        "be under photara_rs/bin/. Reinstall the matching wheel or rebuild the "
        "extension for this Python version. Original error: "
        f"{exc}"
    ) from exc


def resource_dir() -> str:
    """Directory containing the OptiX/CUDA kernel blobs."""
    override = os.environ.get("PHOTARA_RESOURCE_DIR")
    if override:
        return override
    if _IS_DEV_TREE and os.path.isdir(os.path.join(_DEV_INSTALL_DIR, "lib")):
        return _DEV_INSTALL_DIR
    return _PACKAGE_DIR


__all__ = ["_core", "resource_dir"]
