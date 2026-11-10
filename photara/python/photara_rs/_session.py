"""Low-level, handle-based renderer session.

Most users should start with :class:`photara_rs.Scene`. This module mirrors the
C ABI directly and is useful for render farms, custom scene builders and tests.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ._backend import _core, resource_dir
from ._errors import PhotaraClosedError
from ._material import Material, validate_values
from ._math import as_float2, as_float3, as_matrix4

_MAX_INT32 = 2**31 - 1
_MIN_INT32 = -(2**31)


def _as_float(value, name: str, minimum: Optional[float] = None) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _as_color(value, name: str) -> np.ndarray:
    color = as_float3(value, name)
    if np.any(color < 0.0):
        raise ValueError(f"{name} must not contain negative values")
    return color


def _as_size(value, name: str = "size") -> Tuple[float, float]:
    size_u, size_v = as_float2(value, name)
    if size_u <= 0.0 or size_v <= 0.0:
        raise ValueError(f"{name} values must be positive")
    return size_u, size_v


def _as_object_id(value) -> int:
    object_id = int(value)
    if object_id < _MIN_INT32 or object_id > _MAX_INT32:
        raise ValueError("object_id must fit in a signed 32-bit integer")
    if abs(object_id) > 2**24:
        raise ValueError(
            "object_id must satisfy |object_id| <= 2^24 so it survives the "
            "float32 object_id render pass exactly"
        )
    return object_id


def _as_vertices(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if array.shape[0] <= 0:
        raise ValueError("vertices must not be empty")
    if not np.isfinite(array).all():
        raise ValueError("vertices must contain finite values")
    return np.ascontiguousarray(array)


def _as_indices(value, vertex_count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("indices must have shape (T, 3)")
    if array.shape[0] <= 0:
        raise ValueError("indices must not be empty")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("indices must be an integer array")
    minimum = int(array.min())
    maximum = int(array.max())
    if minimum < 0 or maximum >= vertex_count:
        raise ValueError("indices must reference vertices in range [0, N)")
    if maximum > _MAX_INT32:
        raise ValueError("indices must fit in signed 32-bit integers")
    return np.ascontiguousarray(array, dtype=np.int32)


def _as_optional_normals(value, vertex_count: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape != (vertex_count, 3):
        raise ValueError("normals must have shape (N, 3) matching vertices")
    if not np.isfinite(array).all():
        raise ValueError("normals must contain finite values")
    return np.ascontiguousarray(array)


def _as_optional_uvs(value, vertex_count: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape != (vertex_count, 2):
        raise ValueError("uvs must have shape (N, 2) matching vertices")
    if not np.isfinite(array).all():
        raise ValueError("uvs must contain finite values")
    return np.ascontiguousarray(array)


class Session:
    """A low-level handle-based OptiX render session.

    A session owns one scene and one GPU context. Geometry and lights are
    referred to by integer handles. The session can be rendered repeatedly and
    supports object/material/light mutation between frames.

    Most users should use :class:`photara_rs.Scene`, which provides a
    Blender-style object/data-block API on top of this class. A session holds
    the GPU context for its whole lifetime; use it as a context manager or call
    :meth:`close` explicitly.

    Methods that touch the renderer are not safe to call concurrently with
    :meth:`render` on the same session. Rendering releases the GIL, so separate
    sessions can render from separate Python threads.
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
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        samples = int(samples)
        if samples <= 0:
            raise ValueError("samples must be positive")
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        max_bounces = int(max_bounces)
        if max_bounces <= 0:
            raise ValueError("max_bounces must be positive")

        self.width_ = width
        self.height_ = height
        self._samples = samples
        self._seed = seed
        self._max_bounces = max_bounces
        self._denoise = bool(denoise)
        self._adaptive_sampling = bool(adaptive_sampling)
        self._device = str(device)
        self._material_values: list[dict] = []
        self._material_objects: dict[int, Material] = {}
        self._context = None

        self._context = _core.Context(
            width,
            height,
            samples,
            seed,
            max_bounces,
            self._denoise,
            self._adaptive_sampling,
            self._device,
            resource_dir(),
        )

    # ------------------------------------------------------------------
    # Session state.
    # ------------------------------------------------------------------
    @property
    def _ctx(self):
        if self._context is None:
            raise PhotaraClosedError("Session is closed")
        return self._context

    @property
    def closed(self) -> bool:
        return self._context is None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Release the GPU context. All later renderer calls raise."""
        self._context = None

    def __repr__(self) -> str:
        if self.closed:
            return "Session(closed)"
        return f"Session({self.width}x{self.height}, device={self.device_name!r})"

    # ------------------------------------------------------------------
    # Materials.
    # ------------------------------------------------------------------
    def add_material(self, material: Optional[Material] = None, **kwargs) -> int:
        """Add a Principled BSDF material and return its integer index.

        Either pass a :class:`Material`, or keyword arguments matching its
        fields. A passed Material is attached to this session, so later property
        changes update the renderer automatically.
        """
        ctx = self._ctx
        if material is None:
            material = Material(**kwargs)
        elif kwargs:
            raise TypeError("Pass either a Material or keyword arguments, not both")
        if not isinstance(material, Material):
            raise TypeError("material must be a photara_rs.Material")
        if material.bound:
            raise ValueError("Material is already attached to a Session or Scene")

        index = int(ctx.add_material(*material._as_args()))
        material._bind(self, index)
        self._material_values.append(material.to_dict())
        self._material_objects[index] = material
        return index

    def update_material(
        self,
        index: int,
        material: Optional[Material] = None,
        **kwargs,
    ) -> None:
        """Update material fields in place.

        With only keyword arguments, unspecified fields keep their current
        values. Passing a Material replaces the full parameter set.
        """
        self._ctx
        index = int(index)
        if material is not None and kwargs:
            raise TypeError("Pass either a Material or keyword arguments, not both")
        if material is not None:
            if not isinstance(material, Material):
                raise TypeError("material must be a photara_rs.Material")
            if material.bound and material._session is not self:
                raise ValueError("Material is attached to another Session or Scene")
            values = material.to_dict()
        else:
            if index < 0 or index >= len(self._material_values):
                raise IndexError(f"Material index {index} is out of range")
            values = dict(self._material_values[index])
            values.update(kwargs)
            values = validate_values(values)
        self._update_material_values(index, values)
        if material is not None:
            old = self._material_objects.get(index)
            if old is not None and old is not material:
                old._unbind()
            material._bind(self, index)
            self._material_objects[index] = material

    def _update_material_values(self, index: int, values: dict) -> None:
        ctx = self._ctx
        index = int(index)
        if index < 0 or index >= len(self._material_values):
            raise IndexError(f"Material index {index} is out of range")
        values = validate_values(values)
        ctx.update_material(index, *Material(**values)._as_args())
        self._material_values[index] = values
        bound = self._material_objects.get(index)
        if bound is not None:
            bound._set_values(values, sync=False)

    # ------------------------------------------------------------------
    # Meshes and objects.
    # ------------------------------------------------------------------
    def add_mesh(
        self,
        vertices,
        indices,
        normals=None,
        uvs=None,
        transform=None,
        material: int = -1,
        object_id: int = 0,
    ) -> int:
        """Add a triangle mesh object and return its object handle."""
        ctx = self._ctx
        vertices_array = _as_vertices(vertices)
        indices_array = _as_indices(indices, int(vertices_array.shape[0]))
        normals_array = _as_optional_normals(normals, int(vertices_array.shape[0]))
        uvs_array = _as_optional_uvs(uvs, int(vertices_array.shape[0]))
        matrix_array = None if transform is None else as_matrix4(transform, "transform")
        material = int(material)
        if material < -1 or material >= len(self._material_values):
            raise IndexError(f"Material index {material} is out of range")

        return int(
            ctx.add_mesh(
                vertices_array,
                indices_array,
                normals_array,
                uvs_array,
                matrix_array,
                material,
                _as_object_id(object_id),
            )
        )

    def add_instance(self, source_object: int, transform=None, object_id: int = 0) -> int:
        """Add an object sharing the mesh of ``source_object``."""
        ctx = self._ctx
        source_object = int(source_object)
        if source_object < 0:
            raise ValueError("source_object must be a non-negative object handle")
        matrix_array = None if transform is None else as_matrix4(transform, "transform")
        return int(ctx.add_instance(source_object, matrix_array, _as_object_id(object_id)))

    def set_object_transform(self, object_handle: int, transform) -> None:
        self._ctx.set_object_transform(int(object_handle), as_matrix4(transform, "transform"))

    def set_object_visible(self, object_handle: int, visible: bool = True) -> None:
        self._ctx.set_object_visible(int(object_handle), bool(visible))

    def set_object_id(self, object_handle: int, object_id: int) -> None:
        self._ctx.set_object_id(int(object_handle), _as_object_id(object_id))

    def set_object_material(self, object_handle: int, material: int) -> None:
        material = int(material)
        if material < -1 or material >= len(self._material_values):
            raise IndexError(f"Material index {material} is out of range")
        self._ctx.set_object_material(int(object_handle), material)

    def remove_object(self, object_handle: int) -> None:
        self._ctx.remove_object(int(object_handle))

    def clear_objects(self) -> None:
        self._ctx.clear_objects()

    # ------------------------------------------------------------------
    # Camera.
    # ------------------------------------------------------------------
    def set_camera(
        self,
        position: Sequence[float],
        target: Sequence[float],
        up: Optional[Sequence[float]] = (0.0, 1.0, 0.0),
        fov: float = 45.0,
        near_clip: float = 0.001,
        far_clip: float = 1000.0,
    ) -> None:
        """Place the perspective camera by look-at."""
        self._ctx.set_camera(
            as_float3(position, "position"),
            as_float3(target, "target"),
            None if up is None else as_float3(up, "up"),
            _as_float(fov, "fov", 1e-6),
            _as_float(near_clip, "near_clip", 1e-6),
            _as_float(far_clip, "far_clip", 1e-6),
        )

    def set_camera_matrix(
        self,
        matrix,
        fov: float = 45.0,
        near_clip: float = 0.001,
        far_clip: float = 1000.0,
    ) -> None:
        """Set the camera from a 4x4 transform; local +Z is the view direction."""
        self._ctx.set_camera_matrix(
            as_matrix4(matrix, "matrix"),
            _as_float(fov, "fov", 1e-6),
            _as_float(near_clip, "near_clip", 1e-6),
            _as_float(far_clip, "far_clip", 1e-6),
        )

    # ------------------------------------------------------------------
    # Lights.
    # ------------------------------------------------------------------
    def add_area_light(
        self,
        position: Sequence[float],
        target: Sequence[float],
        size=(1.0, 1.0),
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 10.0,
        up: Optional[Sequence[float]] = (0.0, 1.0, 0.0),
    ) -> int:
        """Add a rectangular area light and return its light handle."""
        size_u, size_v = _as_size(size, "size")
        return int(
            self._ctx.add_area_light(
                as_float3(position, "position"),
                as_float3(target, "target"),
                None if up is None else as_float3(up, "up"),
                size_u,
                size_v,
                _as_color(color, "color"),
                _as_float(strength, "strength", 0.0),
            )
        )

    def update_area_light(
        self,
        light: int,
        position: Sequence[float],
        target: Sequence[float],
        size=(1.0, 1.0),
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 10.0,
        up: Optional[Sequence[float]] = (0.0, 1.0, 0.0),
    ) -> None:
        size_u, size_v = _as_size(size, "size")
        self._ctx.update_area_light(
            int(light),
            as_float3(position, "position"),
            as_float3(target, "target"),
            None if up is None else as_float3(up, "up"),
            size_u,
            size_v,
            _as_color(color, "color"),
            _as_float(strength, "strength", 0.0),
        )

    def add_sun_light(
        self,
        direction: Sequence[float],
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 1.0,
        angle: float = 0.0,
    ) -> int:
        """Add a sun light and return its light handle."""
        return int(
            self._ctx.add_sun_light(
                as_float3(direction, "direction"),
                _as_color(color, "color"),
                _as_float(strength, "strength", 0.0),
                _as_float(angle, "angle", 0.0),
            )
        )

    def update_sun_light(
        self,
        light: int,
        direction: Sequence[float],
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 1.0,
        angle: float = 0.0,
    ) -> None:
        self._ctx.update_sun_light(
            int(light),
            as_float3(direction, "direction"),
            _as_color(color, "color"),
            _as_float(strength, "strength", 0.0),
            _as_float(angle, "angle", 0.0),
        )

    def set_light_transform(self, light: int, transform) -> None:
        self._ctx.set_light_transform(int(light), as_matrix4(transform, "transform"))

    def set_light_visible(self, light: int, visible: bool = True) -> None:
        self._ctx.set_light_visible(int(light), bool(visible))

    def remove_light(self, light: int) -> None:
        self._ctx.remove_light(int(light))

    def clear_lights(self) -> None:
        self._ctx.clear_lights()

    def clear_scene(self) -> None:
        self._ctx.clear_scene()

    def set_background(
        self,
        color: Sequence[float] = (0.05, 0.05, 0.05),
        strength: float = 1.0,
    ) -> None:
        """Set a constant-color world background."""
        self._ctx.set_background(
            _as_color(color, "color"),
            _as_float(strength, "strength", 0.0),
        )

    # ------------------------------------------------------------------
    # Rendering.
    # ------------------------------------------------------------------
    def set_seed(self, seed: int) -> None:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self._ctx.set_seed(seed)
        self._seed = seed

    def set_samples(self, samples: int) -> None:
        samples = int(samples)
        if samples <= 0:
            raise ValueError("samples must be positive")
        self._ctx.set_samples(samples)
        self._samples = samples

    def render(self, seed: Optional[int] = None) -> None:
        """Render synchronously. ``seed`` is applied before rendering."""
        ctx = self._ctx
        if seed is not None:
            seed = int(seed)
            if seed < 0:
                raise ValueError("seed must be non-negative")
            ctx.set_seed(seed)
            self._seed = seed
        ctx.render()

    def get_pass(self, name: str = "combined", flip: bool = True) -> np.ndarray:
        """Return a render pass as a top-down ``(H, W, C)`` float32 array."""
        array = self._ctx.get_pass(str(name))
        if flip:
            array = np.flipud(array)
        return array

    def beauty(self, flip: bool = True) -> np.ndarray:
        """Linear RGBA combined pass, shape ``(H, W, 4)``."""
        return self.get_pass("combined", flip=flip)

    def depth(self, flip: bool = True) -> np.ndarray:
        """Depth pass, shape ``(H, W)``. Misses contain ``1e10``."""
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

    # ------------------------------------------------------------------
    # Metadata.
    # ------------------------------------------------------------------
    @property
    def width(self) -> int:
        return self.width_ if self.closed else int(self._ctx.width)

    @property
    def height(self) -> int:
        return self.height_ if self.closed else int(self._ctx.height)

    @property
    def samples(self) -> int:
        return self._samples

    @samples.setter
    def samples(self, value: int) -> None:
        self.set_samples(value)

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self.set_seed(value)

    @property
    def max_bounces(self) -> int:
        return self._max_bounces

    @property
    def denoise(self) -> bool:
        return self._denoise

    @property
    def adaptive_sampling(self) -> bool:
        return self._adaptive_sampling

    @property
    def device_name(self) -> str:
        return str(self._ctx.device_name)

    @property
    def last_error(self) -> str:
        return "" if self.closed else str(self._ctx.last_error)


def _linear_to_srgb_values(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, None)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Convert the colour channels of a linear image to sRGB.

    Alpha is preserved unchanged for RGBA input.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3 and array.shape[2] >= 3:
        result = array.copy()
        result[..., :3] = _linear_to_srgb_values(array[..., :3])
        return result
    return _linear_to_srgb_values(array)


def save_png(path: str, image: np.ndarray, srgb: bool = True) -> None:
    """Save a 2D, RGB or RGBA float image as an 8-bit PNG.

    Requires Pillow. Alpha is preserved for RGBA input; colour values are
    clipped to [0, 1] after the optional sRGB transform.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ImportError(
            "save_png requires Pillow. Install it with "
            "`pip install photara-rs[png]` or `pip install Pillow`."
        ) from exc

    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError("image must have shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4)")

    if array.shape[2] == 1:
        gray = array[..., 0]
        if srgb:
            gray = _linear_to_srgb_values(gray)
        gray = np.clip(gray, 0.0, 1.0)
        Image.fromarray((gray * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)
        return

    rgb = array[..., :3]
    if srgb:
        rgb = _linear_to_srgb_values(rgb)
    rgb = np.clip(rgb, 0.0, 1.0)

    if array.shape[2] == 4:
        rgba = np.empty((array.shape[0], array.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = (rgb * 255.0 + 0.5).astype(np.uint8)
        rgba[..., 3] = np.clip(array[..., 3] * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(rgba, mode="RGBA").save(path)
    else:
        Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


__all__ = ["Session", "linear_to_srgb", "save_png"]
