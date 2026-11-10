"""Texture configuration validation shared by Material and World."""

from __future__ import annotations

import os
from typing import Any, Mapping

_COLORSPACES = {
    "auto": "auto",
    "srgb": "sRGB",
    "scene_linear": "scene_linear",
    "linear": "scene_linear",
    "data": "data",
}

_INTERPOLATIONS = {
    "linear": 0,
    "closest": 1,
    "cubic": 2,
    "smart": 3,
}

_EXTENSIONS = {
    "repeat": 0,
    "extend": 1,
    "clip": 2,
    "mirror": 3,
}

_CONFIG_KEYS = {"path", "filename", "colorspace", "interpolation", "extension"}


def normalize_texture_config(
    value: Any,
    *,
    default_colorspace: str = "sRGB",
    default_interpolation: str = "linear",
    default_extension: str = "repeat",
) -> dict:
    """Return a validated texture config with absolute path and enum indices."""
    if isinstance(value, (str, os.PathLike)):
        data = {"path": value}
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        raise TypeError("texture must be a path or a mapping")

    unknown = set(data) - _CONFIG_KEYS
    if unknown:
        raise TypeError(f"Unknown texture fields: {', '.join(sorted(unknown))}")

    path_value = data.get("path", data.get("filename"))
    if path_value is None:
        raise ValueError("texture path is required")
    path = os.path.abspath(os.fspath(path_value))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"texture file does not exist: {path}")

    colorspace_value = str(data.get("colorspace", default_colorspace)).strip()
    colorspace_key = colorspace_value.lower()
    if colorspace_key not in _COLORSPACES:
        raise ValueError(
            f"Unknown texture colorspace {colorspace_value!r}; "
            "expected auto, sRGB, scene_linear or data"
        )

    interpolation_value = str(data.get("interpolation", default_interpolation)).lower()
    if interpolation_value not in _INTERPOLATIONS:
        raise ValueError(
            f"Unknown texture interpolation {interpolation_value!r}; "
            "expected linear, closest, cubic or smart"
        )

    extension_value = str(data.get("extension", default_extension)).lower()
    if extension_value not in _EXTENSIONS:
        raise ValueError(
            f"Unknown texture extension {extension_value!r}; "
            "expected repeat, extend, clip or mirror"
        )

    return {
        "path": path,
        "colorspace": _COLORSPACES[colorspace_key],
        "interpolation": interpolation_value,
        "extension": extension_value,
        "interpolation_index": _INTERPOLATIONS[interpolation_value],
        "extension_index": _EXTENSIONS[extension_value],
    }


__all__ = ["normalize_texture_config"]
