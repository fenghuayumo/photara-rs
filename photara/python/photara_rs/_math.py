"""Small transform helpers shared by the Scene and Session layers."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

_EPS = 1e-7


def as_float3(value: Sequence[float], name: str = "value") -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return np.ascontiguousarray(array)


def as_float2(value: Sequence[float], name: str = "value") -> Tuple[float, float]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (2,):
        raise ValueError(f"{name} must contain exactly 2 values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return float(array[0]), float(array[1])


def as_matrix4(value, name: str = "matrix") -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return np.ascontiguousarray(array)


def normalize(value: Sequence[float], name: str = "vector") -> np.ndarray:
    vector = as_float3(value, name)
    length = float(np.linalg.norm(vector))
    if length <= _EPS:
        raise ValueError(f"{name} must be non-zero")
    return (vector / length).astype(np.float32)


def basis_matrix(
    right: Sequence[float],
    up: Sequence[float],
    forward: Sequence[float],
    origin: Sequence[float],
) -> np.ndarray:
    """Matrix with the basis vectors in columns and translation in the last column."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 0] = as_float3(right, "right")
    matrix[:3, 1] = as_float3(up, "up")
    matrix[:3, 2] = as_float3(forward, "forward")
    matrix[:3, 3] = as_float3(origin, "origin")
    return matrix


def look_at_components(
    position: Sequence[float],
    target: Sequence[float],
    up: Sequence[float] = (0.0, 1.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right/up/forward for a camera-style +Z view direction."""
    origin = as_float3(position, "position")
    forward = normalize(as_float3(target, "target") - origin, "view direction")
    hint = normalize(up, "up")

    if abs(float(np.dot(hint, forward))) > 0.9999:
        hint = (
            np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(float(forward[2])) < 0.9
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
    right = normalize(np.cross(hint, forward), "right")
    up_vector = np.cross(forward, right).astype(np.float32)
    return right, up_vector, forward


def camera_matrix(
    position: Sequence[float],
    target: Sequence[float],
    up: Sequence[float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    right, up_vector, forward = look_at_components(position, target, up)
    return basis_matrix(right, up_vector, forward, as_float3(position, "position"))


def area_light_matrix(
    position: Sequence[float],
    target: Sequence[float],
    up: Sequence[float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Area-light local +Z points from the target towards the light."""
    origin = as_float3(position, "position")
    forward = normalize(origin - as_float3(target, "target"), "light direction")
    hint = normalize(up, "up")
    if abs(float(np.dot(hint, forward))) > 0.9999:
        hint = (
            np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(float(forward[2])) < 0.9
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
    right = normalize(np.cross(hint, forward), "right")
    up_vector = np.cross(forward, right).astype(np.float32)
    return basis_matrix(right, up_vector, forward, origin)


def sun_light_matrix(direction: Sequence[float]) -> np.ndarray:
    """Sun light local -Z is its travel direction."""
    forward = normalize(-as_float3(direction, "direction"), "sun direction")
    hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(hint, forward))) > 0.9999:
        hint = (
            np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(float(forward[2])) < 0.9
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
    right = normalize(np.cross(hint, forward), "right")
    up_vector = np.cross(forward, right).astype(np.float32)
    return basis_matrix(right, up_vector, forward, (0.0, 0.0, 0.0))


def euler_to_matrix(euler: Sequence[float]) -> np.ndarray:
    """XYZ Euler (radians) to a 3x3 rotation matrix for column vectors."""
    rx, ry, rz = (float(value) for value in as_float3(euler, "rotation_euler"))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float32,
    )


def matrix_to_euler(rotation: np.ndarray) -> np.ndarray:
    """3x3 rotation to XYZ Euler (radians), inverse of :func:`euler_to_matrix`."""
    rotation = np.asarray(rotation, dtype=np.float32)
    sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if sy > _EPS:
        rx = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        ry = math.atan2(float(-rotation[2, 0]), sy)
        rz = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        rx = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        ry = math.atan2(float(-rotation[2, 0]), sy)
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float32)


def decompose(matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an affine matrix into location, XYZ Euler and scale."""
    matrix = as_matrix4(matrix)
    location = matrix[:3, 3].copy()
    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0).astype(np.float32)
    if np.any(scale <= _EPS):
        raise ValueError("matrix has zero scale and cannot be decomposed")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3, dtype=np.float32), atol=1e-5):
        raise ValueError("matrix contains shear and cannot be decomposed into rotation/scale")
    return location, matrix_to_euler(rotation), scale


def compose(location, rotation_euler, scale) -> np.ndarray:
    """Build a 4x4 matrix from location, XYZ Euler (radians) and scale."""
    origin = as_float3(location, "location")
    euler = as_float3(rotation_euler, "rotation_euler")
    scale_array = np.asarray(scale, dtype=np.float32)
    if scale_array.ndim == 0:
        scale_array = np.full(3, float(scale_array), dtype=np.float32)
    scale_array = scale_array.reshape(-1)
    if scale_array.shape != (3,):
        raise ValueError("scale must be a scalar or contain exactly 3 values")
    if not np.isfinite(scale_array).all():
        raise ValueError("scale must contain finite values")

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = euler_to_matrix(euler) * scale_array[np.newaxis, :]
    matrix[:3, 3] = origin
    return matrix


def column(matrix: np.ndarray, index: int) -> np.ndarray:
    """Return a normalized copy of a matrix column (used for light/camera axes)."""
    value = np.asarray(matrix, dtype=np.float32)[:3, index]
    length = float(np.linalg.norm(value))
    if length <= _EPS:
        raise ValueError("matrix basis column is zero")
    return (value / length).astype(np.float32)


__all__ = [
    "area_light_matrix",
    "as_float2",
    "as_float3",
    "as_matrix4",
    "basis_matrix",
    "camera_matrix",
    "column",
    "compose",
    "decompose",
    "euler_to_matrix",
    "look_at_components",
    "matrix_to_euler",
    "normalize",
    "sun_light_matrix",
]
