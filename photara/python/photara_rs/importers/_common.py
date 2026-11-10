"""Shared helpers for the pure-Python importers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .._math import compose


@dataclass
class ImportResult:
    """Objects and data-blocks created by an importer."""

    objects: list = field(default_factory=list)
    meshes: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: Optional[str] = None
    texture_cache_dir: Optional[str] = None


def import_transform(location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=1.0):
    """Build the top-level object transform for an import."""
    return compose(location, rotation, scale)


def compute_vertex_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Area-weighted smooth vertex normals for indexed triangles."""
    vertices = np.asarray(vertices, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int32)
    normals = np.zeros_like(vertices, dtype=np.float32)
    if indices.size == 0:
        return normals

    tri = vertices[indices]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, indices[:, corner], face_normals)

    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, np.newaxis]
    normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


def srgb_to_linear(value) -> np.ndarray:
    """Convert sRGB-encoded channel values to linear values."""
    values = np.asarray(value, dtype=np.float32)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


__all__ = [
    "ImportResult",
    "compute_vertex_normals",
    "import_transform",
    "srgb_to_linear",
]
