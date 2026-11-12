"""Low-level, handle-based renderer session.

Most users should start with :class:`photara_rs.Scene`. This module mirrors the
C ABI directly and is useful for render farms, custom scene builders and tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ._backend import _core, resource_dir
from ._errors import (
    PhotaraCancelledError,
    PhotaraClosedError,
    PhotaraRenderError,
)
from ._material import Material, validate_values
from ._texture import normalize_texture_config
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


@dataclass(frozen=True)
class RenderProgress:
    """Snapshot of an asynchronous render job."""

    state: str
    progress: float
    current_sample: int
    num_samples: int
    status: str
    substatus: str


class RenderJob:
    """A non-blocking render started by :meth:`Session.render_async`."""

    _STATE_NAMES = {
        0: "running",
        1: "done",
        2: "cancelled",
        3: "failed",
    }

    def __init__(self, session: "Session", job_id: int, seed: int) -> None:
        self._session = session
        self.id = int(job_id)
        self.seed = int(seed)
        self._snapshot: Optional[RenderProgress] = None
        self._result = None

    def _poll_raw(self) -> RenderProgress:
        ctx = self._session._ctx
        try:
            raw = ctx.render_poll(self.id)
        except RuntimeError as exc:
            raise PhotaraRenderError(str(exc)) from exc
        state = self._STATE_NAMES.get(int(raw["state"]), "failed")
        self._snapshot = RenderProgress(
            state=state,
            progress=float(raw["progress"]),
            current_sample=int(raw["current_sample"]),
            num_samples=int(raw["num_samples"]),
            status=str(raw["status"]),
            substatus=str(raw["substatus"]),
        )
        return self._snapshot

    def poll(self) -> RenderProgress:
        """Return a fresh status snapshot without blocking."""
        return self._poll_raw()

    def _finish(self) -> None:
        ctx = self._session._ctx
        try:
            ctx.render_finish(self.id)
        except RuntimeError as exc:
            raise PhotaraRenderError(str(exc)) from exc

    def _collect(self):
        if self._result is None:
            from .scene import RenderResult

            self._result = RenderResult(self._session, self.seed)
        return self._result

    @property
    def state(self) -> str:
        return self.poll().state

    @property
    def progress(self) -> float:
        return self.poll().progress

    @property
    def current_sample(self) -> int:
        return self.poll().current_sample

    @property
    def num_samples(self) -> int:
        return self.poll().num_samples

    @property
    def status(self) -> str:
        return self.poll().status

    @property
    def substatus(self) -> str:
        return self.poll().substatus

    @property
    def done(self) -> bool:
        return self.state == "done"

    @property
    def cancelled(self) -> bool:
        return self.state == "cancelled"

    @property
    def failed(self) -> bool:
        return self.state == "failed"

    def request_cancel(self) -> None:
        """Request cancellation immediately; keep polling until terminal before editing."""
        if self._result is None:
            self._session._ctx.render_request_cancel(self.id)

    def preview_update(self, revision=0):
        """Return (revision, frame) when the running job has newer native pixels."""
        self.poll()  # Validate that this job still owns the session.
        raw = self._session._ctx.preview(int(revision))
        if raw is None:
            return None
        from types import SimpleNamespace
        revision, rgba, depth = raw
        return revision, SimpleNamespace(beauty=np.flipud(rgba), depth=np.flipud(depth),
            depth_valid=np.flipud(np.isfinite(depth) & (depth < 1e9)))

    def preview(self):
        """Read only combined and depth from a completed job, without collecting AOVs."""
        snapshot = self.poll()
        if snapshot.state != "done":
            raise PhotaraRenderError("Preview is only available for a completed render")
        self._finish()
        from types import SimpleNamespace
        beauty = self._session.beauty()
        depth = self._session.depth()
        return SimpleNamespace(beauty=beauty, depth=depth, depth_valid=np.isfinite(depth) & (depth < 1e9))

    def cancel(self) -> None:
        """Cancel the render and wait for the renderer to stop."""
        if self._result is not None:
            return
        snapshot = self.poll()
        if snapshot.state != "running":
            return
        ctx = self._session._ctx
        try:
            ctx.render_cancel(self.id)
        except RuntimeError as exc:
            raise PhotaraRenderError(str(exc)) from exc
        self.poll()

    def wait(self, timeout: Optional[float] = None, poll_interval: float = 0.02):
        """Block until the job finishes and return a :class:`RenderResult`.

        Raises :class:`PhotaraCancelledError` when cancelled,
        :class:`PhotaraRenderError` on failure, and :class:`TimeoutError` if
        ``timeout`` expires.
        """
        if self._result is not None:
            return self._result
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            snapshot = self.poll()
            if snapshot.state == "done":
                self._finish()
                return self._collect()
            if snapshot.state == "cancelled":
                raise PhotaraCancelledError(snapshot.status)
            if snapshot.state == "failed":
                raise PhotaraRenderError(snapshot.status)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("render job timed out")
            time.sleep(max(float(poll_interval), 0.001))

    def result(self):
        """Return the completed result; raise if it is not ready."""
        if self._result is not None:
            return self._result
        snapshot = self.poll()
        if snapshot.state == "done":
            self._finish()
            return self._collect()
        if snapshot.state == "cancelled":
            raise PhotaraCancelledError(snapshot.status)
        if snapshot.state == "failed":
            raise PhotaraRenderError(snapshot.status)
        raise PhotaraRenderError("render job is still running")

    def __repr__(self) -> str:
        state = self._snapshot.state if self._snapshot is not None else "running"
        progress = self._snapshot.progress if self._snapshot is not None else 0.0
        return f"RenderJob(id={self.id}, state={state!r}, progress={progress:.3f})"


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
        denoiser: str = "AUTO",
        denoise_passes: str = "RGB_ALBEDO_NORMAL",
        denoise_prefilter: str = "ACCURATE",
        denoise_quality: str = "HIGH",
        denoise_use_gpu: bool = True,
        denoise_upscale_quality: str = "NONE",
        viewport: bool = False,
        render_mode: str = "REFERENCE",
        samples_per_frame: int = 1,
    ) -> None:
        try:
            factor = {"NONE": 1.0, "QUALITY": 1.5, "BALANCED": 1.0 / 0.58, "PERFORMANCE": 2.0, "ULTRA_PERFORMANCE": 3.0}[denoise_upscale_quality]
            denoise_args = (
                {"AUTO": 0, "OPTIX": 2, "OPENIMAGEDENOISE": 4, "DLSS": 8}[denoiser],
                {"RGB": 0, "RGB_ALBEDO": 1, "RGB_ALBEDO_NORMAL": 5}[denoise_passes],
                {"NONE": 1, "FAST": 2, "ACCURATE": 3}[denoise_prefilter],
                {"HIGH": 1, "BALANCED": 2, "FAST": 3}[denoise_quality],
                bool(denoise_use_gpu),
            )
        except KeyError as exc:
            raise ValueError(f"Invalid denoiser setting: {exc.args[0]}") from None
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
        self._active_job: Optional[RenderJob] = None
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
            bool(viewport),
        )
        try:
            self.configure_render_mode(render_mode, samples_per_frame=samples_per_frame)
        except Exception:
            self.close()
            raise
        if self._denoise:
            try:
                self._context.configure_denoiser(*denoise_args)
                self._context.configure_dlss(factor)
            except Exception:
                self.close()
                raise


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
        """Cancel any active render and release the GPU context."""
        job = self._active_job
        if job is not None and self._context is not None:
            try:
                if job.state == "running":
                    job.cancel()
            except Exception:
                # Close must stay robust even if the job was superseded.
                pass
        self._active_job = None
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

    def set_material_base_color_texture(self, material_index: int, texture) -> None:
        """Assign a Base Color image texture to an existing material."""
        config = normalize_texture_config(texture)
        index = int(material_index)
        if index < 0 or index >= len(self._material_values):
            raise IndexError(f"Material index {index} is out of range")
        self._ctx.set_material_base_color_texture(
            index,
            config["path"],
            config["colorspace"],
            config["interpolation_index"],
            config["extension_index"],
        )
        bound = self._material_objects.get(index)
        if bound is not None:
            bound._set_texture_config(config)

    def clear_material_base_color_texture(self, material_index: int) -> None:
        index = int(material_index)
        if index < 0 or index >= len(self._material_values):
            raise IndexError(f"Material index {index} is out of range")
        self._ctx.clear_material_base_color_texture(index)
        bound = self._material_objects.get(index)
        if bound is not None:
            bound._set_texture_config(None)

    # ------------------------------------------------------------------
    # Meshes and objects.
    # ------------------------------------------------------------------
    def add_gaussian_splats(self, positions, scales, rotations=None, colors=None,
                            opacities=None, sh_rest=None, transform=None, object_id=0):
        from .gsplat import validate_gaussians
        p, s, q, c, sh = validate_gaussians(positions, scales, rotations, colors, opacities, sh_rest)
        matrix = np.eye(4, dtype=np.float32) if transform is None else as_matrix4(transform, "transform")
        return self._ctx.add_gaussian_splats(p, s, q, c, sh, matrix, int(object_id))

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

    def update_mesh(
        self,
        object_handle: int,
        positions,
        normals=None,
        uvs=None,
    ) -> None:
        """Update vertex data without changing topology.

        The object handle may reference any object sharing the mesh; all
        instances of that mesh are updated.
        """
        ctx = self._ctx
        vertices_array = _as_vertices(positions)
        normals_array = _as_optional_normals(normals, int(vertices_array.shape[0]))
        uvs_array = _as_optional_uvs(uvs, int(vertices_array.shape[0]))
        ctx.update_mesh(int(object_handle), vertices_array, normals_array, uvs_array)

    def replace_mesh(
        self,
        object_handle: int,
        positions,
        indices,
        normals=None,
        uvs=None,
        material: int = -1,
    ) -> None:
        """Replace mesh topology and vertex data in place.

        The mesh/object handle stays valid and all instances sharing the mesh
        see the new topology.
        """
        ctx = self._ctx
        vertices_array = _as_vertices(positions)
        indices_array = _as_indices(indices, int(vertices_array.shape[0]))
        normals_array = _as_optional_normals(normals, int(vertices_array.shape[0]))
        uvs_array = _as_optional_uvs(uvs, int(vertices_array.shape[0]))
        material = int(material)
        if material < -1 or material >= len(self._material_values):
            raise IndexError(f"Material index {material} is out of range")
        ctx.replace_mesh(
            int(object_handle),
            vertices_array,
            indices_array,
            normals_array,
            uvs_array,
            material,
        )

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

    def add_point_light(
        self,
        position: Sequence[float],
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 100.0,
        radius: float = 0.0,
    ) -> int:
        """Add a point light (radius 0) or sphere light (radius > 0)."""
        return int(
            self._ctx.add_point_light(
                as_float3(position, "position"),
                _as_color(color, "color"),
                _as_float(strength, "strength", 0.0),
                _as_float(radius, "radius", 0.0),
            )
        )

    def update_point_light(
        self,
        light: int,
        position: Sequence[float],
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 100.0,
        radius: float = 0.0,
    ) -> None:
        self._ctx.update_point_light(
            int(light),
            as_float3(position, "position"),
            _as_color(color, "color"),
            _as_float(strength, "strength", 0.0),
            _as_float(radius, "radius", 0.0),
        )

    def add_spot_light(
        self,
        position: Sequence[float],
        direction: Sequence[float],
        up: Optional[Sequence[float]] = (0.0, 1.0, 0.0),
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 100.0,
        radius: float = 0.0,
        angle: float = 45.0,
        smooth: float = 0.5,
    ) -> int:
        """Add a spot light. ``direction`` is the beam travel direction."""
        angle = _as_float(angle, "angle")
        if angle <= 0.0 or angle >= 180.0:
            raise ValueError("angle must be in (0, 180) degrees")
        smooth = _as_float(smooth, "smooth")
        if smooth < 0.0 or smooth > 1.0:
            raise ValueError("smooth must be in [0, 1]")
        return int(
            self._ctx.add_spot_light(
                as_float3(position, "position"),
                as_float3(direction, "direction"),
                None if up is None else as_float3(up, "up"),
                _as_color(color, "color"),
                _as_float(strength, "strength", 0.0),
                _as_float(radius, "radius", 0.0),
                angle,
                smooth,
            )
        )

    def update_spot_light(
        self,
        light: int,
        position: Sequence[float],
        direction: Sequence[float],
        up: Optional[Sequence[float]] = (0.0, 1.0, 0.0),
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 100.0,
        radius: float = 0.0,
        angle: float = 45.0,
        smooth: float = 0.5,
    ) -> None:
        angle = _as_float(angle, "angle")
        if angle <= 0.0 or angle >= 180.0:
            raise ValueError("angle must be in (0, 180) degrees")
        smooth = _as_float(smooth, "smooth")
        if smooth < 0.0 or smooth > 1.0:
            raise ValueError("smooth must be in [0, 1]")
        self._ctx.update_spot_light(
            int(light),
            as_float3(position, "position"),
            as_float3(direction, "direction"),
            None if up is None else as_float3(up, "up"),
            _as_color(color, "color"),
            _as_float(strength, "strength", 0.0),
            _as_float(radius, "radius", 0.0),
            angle,
            smooth,
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

    def set_background_texture(self, texture) -> None:
        """Use an equirectangular environment image as the world background."""
        config = normalize_texture_config(texture, default_colorspace="scene_linear")
        self._ctx.set_background_texture(
            config["path"],
            config["colorspace"],
            config["interpolation_index"],
            config["extension_index"],
        )

    def clear_background_texture(self) -> None:
        self._ctx.clear_background_texture()

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

    def _job_is_running(self) -> bool:
        job = self._active_job
        if job is None:
            return False
        try:
            return job.state == "running"
        except PhotaraRenderError:
            # The job was superseded by a synchronous render or another job.
            self._active_job = None
            return False

    def configure_render_mode(self, mode="REFERENCE", *, samples_per_frame=1):
        """Select an experimental per-frame baseline; does not enable ReSTIR/GI cache.

        Existing samples, materials and denoiser settings are retained. Calling
        this method restarts the RealTime sample sequence.
        """
        modes = {"REFERENCE": 0, "REALTIME": 1}
        if mode not in modes:
            raise ValueError("render mode must be REFERENCE or REALTIME")
        if not isinstance(samples_per_frame, int) or isinstance(samples_per_frame, bool):
            raise ValueError("samples_per_frame must be an integer in [1, 64]")
        if not 1 <= samples_per_frame <= 64:
            raise ValueError("samples_per_frame must be in [1, 64]")
        self._ctx.set_render_mode(modes[mode], samples_per_frame)

    def render_async(self, seed: Optional[int] = None) -> RenderJob:
        """Start a render without blocking and return a :class:`RenderJob`."""
        ctx = self._ctx
        if self._job_is_running():
            raise PhotaraRenderError(
                "An asynchronous render is already running; cancel or wait first"
            )
        self._active_job = None
        if seed is not None:
            seed = int(seed)
            if seed < 0:
                raise ValueError("seed must be non-negative")
            ctx.set_seed(seed)
            self._seed = seed
        job = RenderJob(self, int(ctx.render_begin()), self._seed)
        self._active_job = job
        return job

    def render(self, seed: Optional[int] = None) -> None:
        """Render synchronously. ``seed`` is applied before rendering."""
        ctx = self._ctx
        if self._job_is_running():
            raise PhotaraRenderError(
                "An asynchronous render is running; cancel or wait first"
            )
        self._active_job = None
        if seed is not None:
            seed = int(seed)
            if seed < 0:
                raise ValueError("seed must be non-negative")
            ctx.set_seed(seed)
            self._seed = seed
        ctx.render()

    def resize(self, width: int, height: int) -> None:
        """Resize an idle session while retaining its scene and device."""
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            raise ValueError("Render resolution must be positive")
        self._ctx.resize(width, height)
        self.width_, self.height_ = width, height

    def configure_sampling(self, *, noise_threshold=0.01, min_samples=0,
                           time_limit=0.0, denoise_start_sample=0):
        """Configure adaptive sampling, time limit in seconds and viewport denoise start."""
        self._ctx.configure_sampling(float(noise_threshold), int(min_samples),
                                     float(time_limit), int(denoise_start_sample))

    def configure_film(self, *, exposure=1.0, transparent=False, transparent_glass=False,
                       roughness_threshold=0.1, filter_type="BLACKMAN_HARRIS", filter_width=1.5):
        """Configure an idle session's film; exposure is a linear multiplier."""
        filters = {"BOX": 0, "GAUSSIAN": 1, "BLACKMAN_HARRIS": 2}
        if filter_type not in filters:
            raise ValueError(f"Unknown pixel filter: {filter_type}")
        self._ctx.configure_film(float(exposure), bool(transparent), bool(transparent_glass),
                                 float(roughness_threshold), filters[filter_type], float(filter_width))

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
