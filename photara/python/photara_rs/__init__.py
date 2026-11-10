"""Photara renderer: a Cycles-derived, OptiX-first path tracer.

The renderer returns linear float32 passes. Use :func:`linear_to_srgb` and
:func:`save_png` for display, or consume the linear values directly in a
simulation pipeline.

Example
-------
>>> import numpy as np
>>> import photara_rs as pr
>>> session = pr.Session(640, 480, samples=64, device="OPTIX")
>>> material = session.add_material(pr.Material(base_color=(0.8, 0.2, 0.1)))
>>> session.add_mesh(vertices, triangles, material=material)
>>> session.set_camera(position=(3, 2, -4), target=(0, 0, 0))
>>> session.add_area_light(position=(2, 4, -2), target=(0, 0, 0), size=(2, 2))
>>> session.render()
>>> rgb = session.beauty()          # (H, W, 4) linear RGBA, top-down
>>> depth = session.depth()         # (H, W) linear distance
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__version__ = "0.1.0"

__all__ = [
    "Material",
    "Session",
    "linear_to_srgb",
    "save_png",
]


_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PACKAGE_DIR, "..", "..", ".."))
_DEV_INSTALL_DIR = os.path.join(_REPO_ROOT, "install")


def _add_dll_directory(path: Optional[str]) -> None:
    if not path or not os.path.isdir(path):
        return
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path)
        except OSError:
            pass


for _candidate in (
    os.environ.get("PHOTARA_DLL_DIR"),
    os.path.join(_PACKAGE_DIR, "bin"),
    _DEV_INSTALL_DIR,
):
    _add_dll_directory(_candidate)

try:
    from . import _photara as _core
except ImportError:  # Development layout with the module on sys.path.
    import _photara as _core  # type: ignore[no-redef]


def _resource_dir() -> str:
    override = os.environ.get("PHOTARA_RESOURCE_DIR")
    if override:
        return override
    if os.path.isdir(os.path.join(_DEV_INSTALL_DIR, "lib")):
        return _DEV_INSTALL_DIR
    return _PACKAGE_DIR


def _vec3(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values")
    return np.ascontiguousarray(array)


def _vec2(value: Sequence[float], name: str) -> Tuple[float, float]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (2,):
        raise ValueError(f"{name} must contain exactly 2 values")
    return float(array[0]), float(array[1])


@dataclass
class Material:
    """Parameters of a Principled BSDF material."""

    base_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    metallic: float = 0.0
    roughness: float = 0.5
    ior: float = 1.5
    transmission: float = 0.0
    emission_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    emission_strength: float = 0.0


class Session:
    """A single scene and OptiX render session.

    A session owns its scene. Build the scene first, then call :meth:`render`
    and read passes. Creating several sessions is supported but each one holds
    its own device context and GPU memory.
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 768,
        samples: int = 64,
        seed: int = 0,
        max_bounces: int = 8,
        denoise: bool = True,
        adaptive_sampling: bool = True,
        device: str = "AUTO",
    ) -> None:
        self._context = _core.Context(
            int(width),
            int(height),
            int(samples),
            int(seed),
            int(max_bounces),
            bool(denoise),
            bool(adaptive_sampling),
            str(device),
            _resource_dir(),
        )

    # ------------------------------------------------------------------
    # Scene construction.
    # ------------------------------------------------------------------
    def add_material(self, material: Optional[Material] = None, **kwargs) -> int:
        """Add a Principled BSDF material and return its index.

        Either pass a :class:`Material`, or keyword arguments matching its
        fields, e.g. ``add_material(base_color=(1, 0, 0), roughness=0.2)``.
        """
        if material is None:
            material = Material(**kwargs)
        elif kwargs:
            raise TypeError("Pass either a Material or keyword arguments, not both")

        return int(
            self._context.add_material(
                _vec3(material.base_color, "base_color"),
                float(material.metallic),
                float(material.roughness),
                float(material.ior),
                float(material.transmission),
                _vec3(material.emission_color, "emission_color"),
                float(material.emission_strength),
            )
        )

    def add_mesh(
        self,
        vertices,
        indices,
        normals=None,
        uvs=None,
        transform=None,
        material: int = -1,
        object_id: int = 0,
    ) -> None:
        """Add a triangle mesh.

        Parameters
        ----------
        vertices:
            Array of shape ``(N, 3)``, float.
        indices:
            Array of shape ``(T, 3)``, integer triangle indices.
        normals:
            Optional per-vertex normals of shape ``(N, 3)``.
        uvs:
            Optional per-vertex UVs of shape ``(N, 2)``.
        transform:
            Optional 4x4 affine matrix, row-major, column-vector convention
            (translation in the last column).
        material:
            Material index returned by :meth:`add_material`, or ``-1`` for the
            default material.
        object_id:
            Value written to the ``object_id`` pass.
        """
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("indices must have shape (T, 3)")

        normals_array = None
        if normals is not None:
            normals_array = np.ascontiguousarray(normals, dtype=np.float32)
            if normals_array.shape != vertices.shape:
                raise ValueError("normals must have shape (N, 3)")

        uvs_array = None
        if uvs is not None:
            uvs_array = np.ascontiguousarray(uvs, dtype=np.float32)
            if uvs_array.ndim != 2 or uvs_array.shape != (vertices.shape[0], 2):
                raise ValueError("uvs must have shape (N, 2)")

        matrix_array = None
        if transform is not None:
            matrix_array = np.ascontiguousarray(transform, dtype=np.float32)
            if matrix_array.shape != (4, 4):
                raise ValueError("transform must have shape (4, 4)")

        self._context.add_mesh(
            vertices,
            indices,
            normals_array,
            uvs_array,
            matrix_array,
            int(material),
            int(object_id),
        )

    def set_camera(
        self,
        position: Sequence[float],
        target: Sequence[float],
        up: Optional[Sequence[float]] = (0.0, 1.0, 0.0),
        fov: float = 45.0,
        near_clip: float = 0.001,
        far_clip: float = 1000.0,
    ) -> None:
        """Place a perspective camera looking at ``target``."""
        self._context.set_camera(
            _vec3(position, "position"),
            _vec3(target, "target"),
            None if up is None else _vec3(up, "up"),
            float(fov),
            float(near_clip),
            float(far_clip),
        )

    def add_area_light(
        self,
        position: Sequence[float],
        target: Sequence[float],
        size=(1.0, 1.0),
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 10.0,
        up: Optional[Sequence[float]] = (0.0, 1.0, 0.0),
    ) -> None:
        """Add a rectangular area light facing ``target``."""
        size_u, size_v = _vec2(size, "size")
        self._context.add_area_light(
            _vec3(position, "position"),
            _vec3(target, "target"),
            None if up is None else _vec3(up, "up"),
            size_u,
            size_v,
            _vec3(color, "color"),
            float(strength),
        )

    def add_sun_light(
        self,
        direction: Sequence[float],
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 1.0,
        angle: float = 0.0,
    ) -> None:
        """Add a sun light. ``direction`` is the direction the light travels."""
        self._context.add_sun_light(
            _vec3(direction, "direction"),
            _vec3(color, "color"),
            float(strength),
            float(angle),
        )

    def set_background(
        self,
        color: Sequence[float] = (0.05, 0.05, 0.05),
        strength: float = 1.0,
    ) -> None:
        """Set a constant-color world background."""
        self._context.set_background(_vec3(color, "color"), float(strength))

    # ------------------------------------------------------------------
    # Rendering.
    # ------------------------------------------------------------------
    def render(self) -> None:
        """Render synchronously."""
        self._context.render()

    def get_pass(self, name: str = "combined", flip: bool = True) -> np.ndarray:
        """Return a render pass as a top-down ``(H, W, C)`` float32 array.

        Cycles writes render buffers bottom-up; ``flip=True`` (default) makes
        the result match normal image orientation.
        """
        array = self._context.get_pass(str(name))
        if flip:
            array = np.flipud(array)
        return array

    def beauty(self, flip: bool = True) -> np.ndarray:
        """Linear RGBA combined pass, shape ``(H, W, 4)``."""
        return self.get_pass("combined", flip=flip)

    def depth(self, flip: bool = True) -> np.ndarray:
        """Depth pass, shape ``(H, W)``."""
        return self.get_pass("depth", flip=flip)[..., 0]

    def normal(self, flip: bool = True) -> np.ndarray:
        """World-space normal pass, shape ``(H, W, 3)``."""
        return self.get_pass("normal", flip=flip)

    def albedo(self, flip: bool = True) -> np.ndarray:
        """Diffuse color albedo pass, shape ``(H, W, 3)``."""
        return self.get_pass("albedo", flip=flip)

    def object_id(self, flip: bool = True) -> np.ndarray:
        """Object id pass, shape ``(H, W)``, as float32."""
        return self.get_pass("object_id", flip=flip)[..., 0]

    @property
    def width(self) -> int:
        return int(self._context.width)

    @property
    def height(self) -> int:
        return int(self._context.height)

    @property
    def device_name(self) -> str:
        return str(self._context.device_name)

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Release the GPU context. Rendering after close is invalid."""
        if getattr(self, "_context", None) is not None:
            self._context = None

    def __repr__(self) -> str:
        return (
            f"Session({self.width}x{self.height}, device={self.device_name!r})"
        )


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Convert linear RGB values to sRGB display values."""
    values = np.clip(np.asarray(image, dtype=np.float32), 0.0, None)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )


def save_png(path: str, image: np.ndarray, srgb: bool = True) -> None:
    """Save an RGB(A) float image as an 8-bit PNG (requires Pillow)."""
    from PIL import Image

    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError("image must have shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4)")

    if array.shape[2] == 1:
        gray = array[..., 0]
        if srgb:
            gray = linear_to_srgb(gray)
        gray = np.clip(gray, 0.0, 1.0)
        Image.fromarray((gray * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)
        return

    rgb = array[..., :3]
    if srgb:
        rgb = linear_to_srgb(rgb)
    rgb = np.clip(rgb, 0.0, 1.0)
    Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(path)
