"""Blender-style scene graph for Photara.

The public API is intentionally small compared with ``bpy``: a Scene owns
data-block collections (materials, meshes, lights), a camera, a world and a
list of objects. Meshes can be shared by multiple objects; materials and lights
can be updated in place; the same Scene renders repeatedly.

Example
-------
>>> import photara_rs as pr
>>> scene = pr.Scene(1280, 720, samples=64)
>>> red = scene.materials.new("red", base_color=(0.8, 0.2, 0.1))
>>> mesh = scene.meshes.new("sphere", vertices, triangles, material=red)
>>> ball = scene.objects.new("ball", mesh, location=(0, 1, 0))
>>> ball.object_id = 7
>>> scene.camera.location = (4, 3, -6)
>>> scene.camera.target = (0, 1, 0)
>>> scene.lights.new_sun("sun", direction=(-0.4, -1, 0.2))
>>> frame = scene.render(seed=1)
>>> frame.beauty.shape       # (H, W, 4)
>>> frame.save_npz("frame.npz")
"""

from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import numpy as np

from ._errors import PhotaraClosedError, PhotaraError, PhotaraRemovedError
from ._material import Material
from ._math import (
    area_light_matrix,
    as_float2,
    as_float3,
    as_matrix4,
    camera_matrix,
    column,
    compose,
    decompose,
    spot_light_matrix,
    sun_light_matrix,
)
from ._session import (
    RenderJob,
    Session,
    _as_object_id,
    linear_to_srgb,
    save_png,
)
from ._texture import normalize_texture_config


def _readonly(array) -> np.ndarray:
    # Copy first: np.ascontiguousarray may return the caller's array, and we do
    # not want to make the caller's buffer read-only or observe later mutations.
    array = np.array(array, dtype=np.asarray(array).dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _unique_name(items: dict, requested: Optional[str], fallback: str) -> str:
    base = fallback if requested is None or str(requested) == "" else str(requested)
    if base not in items:
        return base
    index = 1
    while f"{base}.{index:03d}" in items:
        index += 1
    return f"{base}.{index:03d}"


class _Collection:
    """Ordered, name-addressable data-block collection."""

    kind = "item"

    def __init__(self, scene: "Scene") -> None:
        self._scene = scene
        self._items: dict[str, object] = {}

    def __iter__(self) -> Iterator:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item) -> bool:
        return getattr(item, "name", item) in self._items

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._items[key]
        if isinstance(key, (int, np.integer)):
            return list(self._items.values())[int(key)]
        raise TypeError(f"{self.kind} collections are indexed by name or integer position")

    def get(self, name: str, default=None):
        return self._items.get(str(name), default)

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def items(self):
        return self._items.items()

    def _unique_name(self, name: Optional[str], fallback: str) -> str:
        return _unique_name(self._items, name, fallback)

    def _add(self, item) -> None:
        self._items[item.name] = item

    def _remove(self, item) -> None:
        if self._items.get(item.name) is item:
            del self._items[item.name]

    def _check_member(self, item) -> None:
        if self._items.get(getattr(item, "name", None)) is not item:
            raise ValueError(f"{self.kind} does not belong to this Scene")

    def __repr__(self) -> str:
        return f"<{self.kind} collection: {len(self._items)} item(s)>"


class _SceneItem:
    def __init__(self, scene: "Scene", name: str) -> None:
        self._scene = scene
        self._name = str(name)
        self._removed = False

    @property
    def scene(self) -> "Scene":
        return self._scene

    @property
    def name(self) -> str:
        return self._name

    def _ensure_alive(self) -> None:
        if self._removed:
            raise PhotaraRemovedError(f"{type(self).__name__} {self.name!r} was removed")
        self._scene._ensure_open()

    def remove(self) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class _TransformItem(_SceneItem):
    def __init__(self, scene: "Scene", name: str, matrix) -> None:
        super().__init__(scene, name)
        self._matrix = as_matrix4(matrix).copy()

    def _apply_matrix(self, matrix: np.ndarray) -> None:
        raise NotImplementedError

    def _set_matrix(self, matrix) -> None:
        self._ensure_alive()
        matrix = as_matrix4(matrix)
        self._apply_matrix(matrix)
        self._matrix = matrix.copy()

    @property
    def matrix_world(self) -> np.ndarray:
        self._ensure_alive()
        return self._matrix.copy()

    @matrix_world.setter
    def matrix_world(self, value) -> None:
        self._set_matrix(value)

    @property
    def location(self) -> np.ndarray:
        self._ensure_alive()
        return self._matrix[:3, 3].copy()

    @location.setter
    def location(self, value) -> None:
        location, rotation, scale = decompose(self._matrix)
        self._set_matrix(compose(as_float3(value, "location"), rotation, scale))

    @property
    def rotation_euler(self) -> np.ndarray:
        """XYZ Euler rotation in radians."""
        self._ensure_alive()
        return decompose(self._matrix)[1]

    @rotation_euler.setter
    def rotation_euler(self, value) -> None:
        location, _, scale = decompose(self._matrix)
        self._set_matrix(compose(location, as_float3(value, "rotation_euler"), scale))

    @property
    def scale(self) -> np.ndarray:
        self._ensure_alive()
        return decompose(self._matrix)[2]

    @scale.setter
    def scale(self, value) -> None:
        location, rotation, _ = decompose(self._matrix)
        self._set_matrix(compose(location, rotation, value))


class Mesh(_SceneItem):
    """A triangle mesh data-block. Multiple objects can share one Mesh."""

    def __init__(
        self,
        scene: "Scene",
        name: str,
        handle: int,
        vertices,
        indices,
        normals,
        uvs,
        material: Optional[Material],
    ) -> None:
        super().__init__(scene, name)
        self._handle = int(handle)
        self._vertices = _readonly(np.asarray(vertices, dtype=np.float32))
        self._indices = _readonly(np.asarray(indices, dtype=np.int32))
        self._normals = (
            _readonly(np.asarray(normals, dtype=np.float32)) if normals is not None else None
        )
        self._uvs = _readonly(np.asarray(uvs, dtype=np.float32)) if uvs is not None else None
        self._material = material
        self._objects: set = set()

    @property
    def vertices(self) -> np.ndarray:
        self._ensure_alive()
        return self._vertices

    @property
    def indices(self) -> np.ndarray:
        self._ensure_alive()
        return self._indices

    @property
    def normals(self) -> Optional[np.ndarray]:
        self._ensure_alive()
        return self._normals

    @property
    def uvs(self) -> Optional[np.ndarray]:
        self._ensure_alive()
        return self._uvs

    @property
    def material(self) -> Optional[Material]:
        self._ensure_alive()
        return self._material

    @material.setter
    def material(self, value: Optional[Material]) -> None:
        self._ensure_alive()
        material = self._scene._resolve_material(value)
        self._scene._session.set_object_material(
            self._handle, -1 if material is None else material.index
        )
        self._material = material

    @property
    def users(self) -> int:
        self._ensure_alive()
        return len(self._objects)

    @property
    def objects(self) -> Tuple["Object", ...]:
        self._ensure_alive()
        return tuple(self._objects)

    def update_positions(self, positions, normals=None, uvs=None) -> None:
        """Update vertex data without changing topology.

        All objects sharing this mesh see the new data. Omitted normals/UVs
        keep their current values.
        """
        self._ensure_alive()
        self._scene._session.update_mesh(
            self._handle, positions, normals=normals, uvs=uvs
        )
        self._vertices = _readonly(np.asarray(positions, dtype=np.float32))
        if normals is not None:
            self._normals = _readonly(np.asarray(normals, dtype=np.float32))
        if uvs is not None:
            self._uvs = _readonly(np.asarray(uvs, dtype=np.float32))

    def update_topology(
        self,
        vertices,
        indices,
        normals=None,
        uvs=None,
        material: Optional[Material] = None,
    ) -> None:
        """Replace topology and vertex data in place.

        ``material=None`` keeps the current material. Supplying no normals/UVs
        removes those attributes, matching a complete mesh replacement.
        """
        self._ensure_alive()
        resolved = None if material is None else self._scene._resolve_material(material)
        self._scene._session.replace_mesh(
            self._handle,
            vertices,
            indices,
            normals=normals,
            uvs=uvs,
            material=-1 if resolved is None else resolved.index,
        )
        self._vertices = _readonly(np.asarray(vertices, dtype=np.float32))
        self._indices = _readonly(np.asarray(indices, dtype=np.int32))
        self._normals = (
            _readonly(np.asarray(normals, dtype=np.float32)) if normals is not None else None
        )
        self._uvs = _readonly(np.asarray(uvs, dtype=np.float32)) if uvs is not None else None
        if resolved is not None:
            self._material = resolved

    def remove(self) -> None:
        self._ensure_alive()
        self._scene.meshes.remove(self)

    def _mark_removed(self) -> None:
        self._removed = True
        self._objects.clear()

    def __repr__(self) -> str:
        return (
            f"Mesh(name={self.name!r}, vertices={len(self._vertices)}, "
            f"triangles={len(self._indices)}, users={self.users})"
        )


class Object(_TransformItem):
    """A mesh object sharing a :class:`Mesh` data-block."""

    def __init__(
        self,
        scene: "Scene",
        name: str,
        handle: int,
        mesh: Mesh,
        matrix,
        object_id: int,
    ) -> None:
        super().__init__(scene, name, matrix)
        self._handle = int(handle)
        self._mesh = mesh
        self._object_id = int(object_id)
        self._visible = True

    @property
    def type(self) -> str:
        return "MESH"

    @property
    def data(self) -> Mesh:
        self._ensure_alive()
        return self._mesh

    @property
    def mesh(self) -> Mesh:
        self._ensure_alive()
        return self._mesh

    @property
    def material(self) -> Optional[Material]:
        """Material of the shared mesh; setting it affects all its objects."""
        return self.data.material

    @material.setter
    def material(self, value: Optional[Material]) -> None:
        self._mesh.material = value

    @property
    def visible(self) -> bool:
        self._ensure_alive()
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._ensure_alive()
        self._scene._session.set_object_visible(self._handle, bool(value))
        self._visible = bool(value)

    @property
    def object_id(self) -> int:
        self._ensure_alive()
        return self._object_id

    @object_id.setter
    def object_id(self, value: int) -> None:
        self._ensure_alive()
        object_id = _as_object_id(value)
        self._scene._session.set_object_id(self._handle, object_id)
        self._object_id = object_id

    def _apply_matrix(self, matrix: np.ndarray) -> None:
        self._scene._session.set_object_transform(self._handle, matrix)

    def remove(self) -> None:
        self._ensure_alive()
        self._scene.objects.remove(self)

    def _mark_removed(self) -> None:
        self._removed = True
        self._mesh._objects.discard(self)
        self._mesh = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        state = "removed" if self._removed else "visible" if self._visible else "hidden"
        return (
            f"Object(name={self.name!r}, data={self._mesh.name if self._mesh else None!r}, "
            f"{state}, object_id={self._object_id})"
        )


class Light(_TransformItem):
    """An area, sun, point or spot light linked to its Scene."""

    def __init__(
        self,
        scene: "Scene",
        name: str,
        light_type: str,
        handle: int,
        matrix,
        params: dict,
    ) -> None:
        super().__init__(scene, name, matrix)
        self._light_type = light_type
        self._handle = int(handle)
        self._params = dict(params)
        self._visible = True

    @property
    def type(self) -> str:
        return "LIGHT"

    @property
    def light_type(self) -> str:
        return self._light_type

    @property
    def visible(self) -> bool:
        self._ensure_alive()
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._ensure_alive()
        self._scene._session.set_light_visible(self._handle, bool(value))
        self._visible = bool(value)

    def _apply_matrix(self, matrix: np.ndarray) -> None:
        self._scene._session.set_light_transform(self._handle, matrix)

    @property
    def color(self) -> tuple[float, float, float]:
        return self._params["color"]

    @color.setter
    def color(self, value) -> None:
        color = as_float3(value, "color")
        if np.any(color < 0.0):
            raise ValueError("color must not contain negative values")
        self._params["color"] = tuple(float(x) for x in color)
        self._sync_params()

    @property
    def strength(self) -> float:
        return self._params["strength"]

    @strength.setter
    def strength(self, value) -> None:
        strength = float(value)
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("strength must be a finite non-negative value")
        self._params["strength"] = strength
        self._sync_params()

    @property
    def radius(self) -> float:
        if self._light_type not in ("POINT", "SPOT"):
            raise AttributeError("radius is only available on point and spot lights")
        return self._params["radius"]

    @radius.setter
    def radius(self, value) -> None:
        if self._light_type not in ("POINT", "SPOT"):
            raise AttributeError("radius is only available on point and spot lights")
        radius = float(value)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("radius must be a finite non-negative value")
        self._params["radius"] = radius
        self._sync_params()

    @property
    def size(self) -> Tuple[float, float]:
        if self._light_type != "AREA":
            raise AttributeError("size is only available on area lights")
        return self._params["size"]

    @size.setter
    def size(self, value) -> None:
        if self._light_type != "AREA":
            raise AttributeError("size is only available on area lights")
        size_u, size_v = as_float2(value, "size")
        if size_u <= 0.0 or size_v <= 0.0:
            raise ValueError("size values must be positive")
        self._params["size"] = (size_u, size_v)
        self._sync_params()

    @property
    def angle(self) -> float:
        if self._light_type != "SUN":
            raise AttributeError("angle is only available on sun lights")
        return self._params["angle"]

    @angle.setter
    def angle(self, value) -> None:
        if self._light_type != "SUN":
            raise AttributeError("angle is only available on sun lights")
        angle = float(value)
        if not np.isfinite(angle) or angle < 0.0:
            raise ValueError("angle must be a finite non-negative value")
        self._params["angle"] = angle
        self._sync_params()

    @property
    def spot_angle(self) -> float:
        if self._light_type != "SPOT":
            raise AttributeError("spot_angle is only available on spot lights")
        return self._params["spot_angle"]

    @spot_angle.setter
    def spot_angle(self, value) -> None:
        if self._light_type != "SPOT":
            raise AttributeError("spot_angle is only available on spot lights")
        angle = float(value)
        if not np.isfinite(angle) or angle <= 0.0 or angle >= 180.0:
            raise ValueError("spot_angle must be in (0, 180) degrees")
        self._params["spot_angle"] = angle
        self._sync_params()

    @property
    def smooth(self) -> float:
        if self._light_type != "SPOT":
            raise AttributeError("smooth is only available on spot lights")
        return self._params["smooth"]

    @smooth.setter
    def smooth(self, value) -> None:
        if self._light_type != "SPOT":
            raise AttributeError("smooth is only available on spot lights")
        smooth = float(value)
        if not np.isfinite(smooth) or smooth < 0.0 or smooth > 1.0:
            raise ValueError("smooth must be in [0, 1]")
        self._params["smooth"] = smooth
        self._sync_params()

    @property
    def target(self) -> np.ndarray:
        if self._light_type != "AREA":
            raise AttributeError("target is only available on area lights")
        self._ensure_alive()
        return self._matrix[:3, 3] - column(self._matrix, 2)

    @target.setter
    def target(self, value) -> None:
        if self._light_type != "AREA":
            raise AttributeError("target is only available on area lights")
        position = self._matrix[:3, 3]
        up = column(self._matrix, 1)
        self._set_matrix(area_light_matrix(position, value, up))

    @property
    def direction(self) -> np.ndarray:
        if self._light_type not in ("SUN", "SPOT"):
            raise AttributeError("direction is only available on sun and spot lights")
        self._ensure_alive()
        return -column(self._matrix, 2)

    @direction.setter
    def direction(self, value) -> None:
        if self._light_type == "SUN":
            self._set_matrix(sun_light_matrix(value))
        elif self._light_type == "SPOT":
            self._set_matrix(
                spot_light_matrix(self.location, value, column(self._matrix, 1))
            )
        else:
            raise AttributeError("direction is only available on sun and spot lights")

    def _sync_params(self) -> None:
        self._ensure_alive()
        session = self._scene._session
        if self._light_type == "AREA":
            position = self._matrix[:3, 3]
            target = position - column(self._matrix, 2)
            size_u, size_v = self._params["size"]
            session.update_area_light(
                self._handle,
                position,
                target,
                size=(size_u, size_v),
                color=self._params["color"],
                strength=self._params["strength"],
                up=column(self._matrix, 1),
            )
        elif self._light_type == "SUN":
            session.update_sun_light(
                self._handle,
                -column(self._matrix, 2),
                color=self._params["color"],
                strength=self._params["strength"],
                angle=self._params["angle"],
            )
        elif self._light_type == "POINT":
            session.update_point_light(
                self._handle,
                self._matrix[:3, 3],
                color=self._params["color"],
                strength=self._params["strength"],
                radius=self._params["radius"],
            )
        else:
            session.update_spot_light(
                self._handle,
                self._matrix[:3, 3],
                -column(self._matrix, 2),
                up=column(self._matrix, 1),
                color=self._params["color"],
                strength=self._params["strength"],
                radius=self._params["radius"],
                angle=self._params["spot_angle"],
                smooth=self._params["smooth"],
            )

    def remove(self) -> None:
        self._ensure_alive()
        self._scene.objects.remove(self)

    def _mark_removed(self) -> None:
        self._removed = True

    def __repr__(self) -> str:
        state = "removed" if self._removed else "enabled" if self._visible else "disabled"
        return f"Light(name={self.name!r}, type={self._light_type!r}, {state})"


class Camera:
    """The scene's perspective camera."""

    def __init__(
        self,
        position: Sequence[float] = (0.0, 0.0, -5.0),
        target: Sequence[float] = (0.0, 0.0, 0.0),
        up: Sequence[float] = (0.0, 1.0, 0.0),
        fov: float = 45.0,
        near_clip: float = 0.001,
        far_clip: float = 1000.0,
        matrix=None,
    ) -> None:
        self._scene: Optional[Scene] = None
        self._matrix = (
            as_matrix4(matrix, "matrix")
            if matrix is not None
            else camera_matrix(position, target, up)
        )
        self._fov = float(fov)
        self._near_clip = float(near_clip)
        self._far_clip = float(far_clip)
        self._validate()

    def _validate(self) -> None:
        if not np.isfinite(self._fov) or self._fov <= 0.0:
            raise ValueError("fov must be a finite positive value")
        if not np.isfinite(self._near_clip) or self._near_clip <= 0.0:
            raise ValueError("near_clip must be positive")
        if not np.isfinite(self._far_clip) or self._far_clip <= self._near_clip:
            raise ValueError("far_clip must be greater than near_clip")

    def _bind(self, scene: "Scene") -> None:
        if self._scene is not None and self._scene is not scene:
            raise ValueError("Camera is already attached to another Scene")
        self._scene = scene
        self._sync()

    def _sync(self) -> None:
        if self._scene is not None:
            self._scene._ensure_open()
            self._scene._session.set_camera_matrix(
                self._matrix, self._fov, self._near_clip, self._far_clip
            )

    def _set_matrix(self, matrix) -> None:
        self._matrix = as_matrix4(matrix)
        self._sync()

    @property
    def matrix_world(self) -> np.ndarray:
        return self._matrix.copy()

    @matrix_world.setter
    def matrix_world(self, value) -> None:
        self._set_matrix(value)

    @property
    def location(self) -> np.ndarray:
        return self._matrix[:3, 3].copy()

    @location.setter
    def location(self, value) -> None:
        location, rotation, scale = decompose(self._matrix)
        self._set_matrix(compose(as_float3(value, "location"), rotation, scale))

    @property
    def rotation_euler(self) -> np.ndarray:
        return decompose(self._matrix)[1]

    @rotation_euler.setter
    def rotation_euler(self, value) -> None:
        location, _, scale = decompose(self._matrix)
        self._set_matrix(compose(location, as_float3(value, "rotation_euler"), scale))

    @property
    def target(self) -> np.ndarray:
        return self._matrix[:3, 3] + column(self._matrix, 2)

    @target.setter
    def target(self, value) -> None:
        self._set_matrix(camera_matrix(self.location, value, (0.0, 1.0, 0.0)))

    @property
    def fov(self) -> float:
        return self._fov

    @fov.setter
    def fov(self, value) -> None:
        self._fov = float(value)
        self._validate()
        self._sync()

    @property
    def near_clip(self) -> float:
        return self._near_clip

    @near_clip.setter
    def near_clip(self, value) -> None:
        self._near_clip = float(value)
        self._validate()
        self._sync()

    @property
    def far_clip(self) -> float:
        return self._far_clip

    @far_clip.setter
    def far_clip(self, value) -> None:
        self._far_clip = float(value)
        self._validate()
        self._sync()

    def __repr__(self) -> str:
        return f"Camera(location={self.location.tolist()}, fov={self.fov})"


class World:
    """Constant-colour world background."""

    def __init__(self, scene: "Scene") -> None:
        self._scene = scene
        self._color = (0.05, 0.05, 0.05)
        self._strength = 1.0
        self._environment_texture = None

    @property
    def environment_texture(self) -> Optional[dict]:
        """Equirectangular environment texture configuration, or None."""
        if self._environment_texture is None:
            return None
        return {
            "path": self._environment_texture["path"],
            "colorspace": self._environment_texture["colorspace"],
            "interpolation": self._environment_texture["interpolation"],
        }

    @environment_texture.setter
    def environment_texture(self, value) -> None:
        self._scene._ensure_open()
        if value is None:
            self._scene._session.clear_background_texture()
            self._environment_texture = None
            return
        config = normalize_texture_config(value, default_colorspace="scene_linear")
        self._scene._session.set_background_texture(value)
        self._environment_texture = config

    @property
    def color(self) -> tuple[float, float, float]:
        return self._color

    @color.setter
    def color(self, value) -> None:
        self._scene._ensure_open()
        color = as_float3(value, "color")
        if np.any(color < 0.0):
            raise ValueError("color must not contain negative values")
        self._color = tuple(float(x) for x in color)
        self._sync()

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value) -> None:
        self._scene._ensure_open()
        strength = float(value)
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("strength must be a finite non-negative value")
        self._strength = strength
        self._sync()

    def _sync(self) -> None:
        self._scene._session.set_background(self._color, self._strength)

    def __repr__(self) -> str:
        return f"World(color={self._color}, strength={self._strength})"


class RenderResult:
    """Passes produced by one render.

    All colour passes are linear float32 and use top-down row order. Depth
    misses contain ``1e10``; use :attr:`depth_valid` as a mask.
    """

    def __init__(self, session: Session, seed: int) -> None:
        self.seed = int(seed)
        self.width = session.width
        self.height = session.height
        self.beauty = session.beauty()
        self.depth = session.depth()
        self.normal = session.normal()
        self.albedo = session.albedo()
        self.object_id = np.rint(session.object_id()).astype(np.int32)
        self.depth_valid = np.isfinite(self.depth) & (self.depth < 1e9)

    @property
    def rgb(self) -> np.ndarray:
        """Linear RGB view of the combined pass."""
        return self.beauty[..., :3]

    @property
    def passes(self) -> dict:
        return {
            "combined": self.beauty,
            "depth": self.depth,
            "depth_valid": self.depth_valid,
            "normal": self.normal,
            "albedo": self.albedo,
            "object_id": self.object_id,
        }

    def save_npz(self, path: str) -> None:
        """Write all passes and the seed to a compressed npz file."""
        np.savez_compressed(
            path,
            combined=self.beauty,
            depth=self.depth,
            depth_valid=self.depth_valid,
            normal=self.normal,
            albedo=self.albedo,
            object_id=self.object_id,
            seed=np.int32(self.seed),
        )

    def save_png(self, path: str, srgb: bool = True) -> None:
        """Save the combined pass as an 8-bit PNG (requires Pillow)."""
        save_png(path, self.beauty, srgb=srgb)

    def __repr__(self) -> str:
        return f"RenderResult({self.width}x{self.height}, seed={self.seed})"


class _MaterialCollection(_Collection):
    kind = "Material"

    def new(self, name: Optional[str] = None, **kwargs) -> Material:
        self._scene._ensure_open()
        unique = self._unique_name(name, "Material")
        material = Material(name=unique, **kwargs)
        self._scene._session.add_material(material)
        self._add(material)
        return material

    def remove(self, material: Material) -> None:
        self._scene._ensure_open()
        self._check_member(material)
        for mesh in self._scene.meshes:
            if mesh.material is material:
                raise PhotaraError(
                    f"Material {material.name!r} is used by mesh {mesh.name!r}"
                )
        self._remove(material)
        material._unbind()


class _MeshCollection(_Collection):
    kind = "Mesh"

    def new(
        self,
        name: Optional[str] = None,
        vertices=None,
        indices=None,
        normals=None,
        uvs=None,
        material: Optional[Material] = None,
    ) -> Mesh:
        self._scene._ensure_open()
        if vertices is None or indices is None:
            raise TypeError("vertices and indices are required")
        unique = self._unique_name(name, "Mesh")
        resolved = self._scene._resolve_material(material)
        handle = self._scene._session.add_mesh(
            vertices,
            indices,
            normals=normals,
            uvs=uvs,
            material=-1 if resolved is None else resolved.index,
            object_id=0,
        )
        # The C ABI creates a mesh + object pair. Keep the prototype hidden and
        # materialize visible objects as instances sharing its geometry.
        self._scene._session.set_object_visible(handle, False)
        mesh = Mesh(
            self._scene,
            unique,
            handle,
            vertices,
            indices,
            normals,
            uvs,
            resolved,
        )
        self._add(mesh)
        return mesh

    def remove(self, mesh: Mesh) -> None:
        self._scene._ensure_open()
        self._check_member(mesh)
        for obj in list(self._scene.objects):
            if isinstance(obj, Object) and obj.mesh is mesh:
                self._scene.objects.remove(obj)
        self._scene._session.remove_object(mesh._handle)
        self._remove(mesh)
        mesh._mark_removed()


class _ObjectCollection(_Collection):
    kind = "Object"

    def new(
        self,
        name: Optional[str] = None,
        data: Optional[Mesh] = None,
        transform=None,
        location=None,
        rotation_euler=None,
        scale=None,
        object_id: int = 0,
    ) -> Object:
        self._scene._ensure_open()
        if not isinstance(data, Mesh):
            raise TypeError("data must be a photara_rs.Mesh")
        data._ensure_alive()
        if data.scene is not self._scene:
            raise ValueError("Mesh belongs to another Scene")
        if transform is not None and any(
            value is not None for value in (location, rotation_euler, scale)
        ):
            raise TypeError(
                "Pass either transform or location/rotation_euler/scale, not both"
            )
        unique = self._unique_name(name, data.name or "Object")
        if transform is not None:
            matrix = as_matrix4(transform, "transform")
        else:
            matrix = compose(
                (0.0, 0.0, 0.0) if location is None else location,
                (0.0, 0.0, 0.0) if rotation_euler is None else rotation_euler,
                (1.0, 1.0, 1.0) if scale is None else scale,
            )
        object_id = _as_object_id(object_id)
        handle = self._scene._session.add_instance(
            data._handle, matrix, object_id
        )
        obj = Object(self._scene, unique, handle, data, matrix, object_id)
        data._objects.add(obj)
        self._add(obj)
        return obj

    def remove(self, obj) -> None:
        self._scene._ensure_open()
        self._check_member(obj)
        if isinstance(obj, Light):
            self._scene._session.remove_light(obj._handle)
            self._scene.lights._remove(obj)
            obj._mark_removed()
        elif isinstance(obj, Object):
            self._scene._session.remove_object(obj._handle)
            obj._mark_removed()
        else:
            raise TypeError("objects collection only contains Object and Light items")
        self._remove(obj)

    def clear(self) -> None:
        self._scene._ensure_open()
        for item in list(self._items.values()):
            self.remove(item)


class _LightCollection(_Collection):
    kind = "Light"

    def _link(self, light: Light) -> Light:
        self._add(light)
        self._scene.objects._add(light)
        return light

    def new_area(
        self,
        name: Optional[str] = None,
        position: Sequence[float] = (0.0, 1.0, 0.0),
        target: Sequence[float] = (0.0, 0.0, 0.0),
        size=(1.0, 1.0),
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 10.0,
        up: Sequence[float] = (0.0, 1.0, 0.0),
    ) -> Light:
        self._scene._ensure_open()
        unique = self._scene.objects._unique_name(name, "Light")
        size_u, size_v = as_float2(size, "size")
        handle = self._scene._session.add_area_light(
            position,
            target,
            size=(size_u, size_v),
            color=color,
            strength=strength,
            up=up,
        )
        matrix = area_light_matrix(position, target, up)
        light = Light(
            self._scene,
            unique,
            "AREA",
            handle,
            matrix,
            {"size": (size_u, size_v), "color": tuple(float(x) for x in as_float3(color, "color")), "strength": float(strength)},
        )
        return self._link(light)

    def new_sun(
        self,
        name: Optional[str] = None,
        direction: Sequence[float] = (0.0, -1.0, 0.0),
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 1.0,
        angle: float = 0.0,
    ) -> Light:
        self._scene._ensure_open()
        unique = self._scene.objects._unique_name(name, "Light")
        handle = self._scene._session.add_sun_light(
            direction, color=color, strength=strength, angle=angle
        )
        matrix = sun_light_matrix(direction)
        light = Light(
            self._scene,
            unique,
            "SUN",
            handle,
            matrix,
            {
                "color": tuple(float(x) for x in as_float3(color, "color")),
                "strength": float(strength),
                "angle": float(angle),
            },
        )
        return self._link(light)

    def new_point(
        self,
        name: Optional[str] = None,
        position: Sequence[float] = (0.0, 1.0, 0.0),
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 100.0,
        radius: float = 0.0,
    ) -> Light:
        self._scene._ensure_open()
        unique = self._scene.objects._unique_name(name, "Light")
        handle = self._scene._session.add_point_light(
            position, color=color, strength=strength, radius=radius
        )
        matrix = compose(position, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        light = Light(
            self._scene,
            unique,
            "POINT",
            handle,
            matrix,
            {
                "color": tuple(float(x) for x in as_float3(color, "color")),
                "strength": float(strength),
                "radius": float(radius),
            },
        )
        return self._link(light)

    def new_spot(
        self,
        name: Optional[str] = None,
        position: Sequence[float] = (0.0, 1.0, 0.0),
        direction: Sequence[float] = (0.0, -1.0, 0.0),
        up: Sequence[float] = (0.0, 1.0, 0.0),
        color: Sequence[float] = (1.0, 1.0, 1.0),
        strength: float = 100.0,
        radius: float = 0.0,
        angle: float = 45.0,
        smooth: float = 0.5,
    ) -> Light:
        self._scene._ensure_open()
        unique = self._scene.objects._unique_name(name, "Light")
        handle = self._scene._session.add_spot_light(
            position,
            direction,
            up=up,
            color=color,
            strength=strength,
            radius=radius,
            angle=angle,
            smooth=smooth,
        )
        matrix = spot_light_matrix(position, direction, up)
        light = Light(
            self._scene,
            unique,
            "SPOT",
            handle,
            matrix,
            {
                "color": tuple(float(x) for x in as_float3(color, "color")),
                "strength": float(strength),
                "radius": float(radius),
                "spot_angle": float(angle),
                "smooth": float(smooth),
            },
        )
        return self._link(light)

    def new(self, name: Optional[str] = None, type: str = "AREA", **kwargs) -> Light:
        light_type = str(type).upper()
        if light_type == "AREA":
            return self.new_area(name=name, **kwargs)
        if light_type == "SUN":
            return self.new_sun(name=name, **kwargs)
        if light_type == "POINT":
            return self.new_point(name=name, **kwargs)
        if light_type == "SPOT":
            return self.new_spot(name=name, **kwargs)
        raise ValueError("type must be 'AREA', 'SUN', 'POINT' or 'SPOT'")

    def remove(self, light: Light) -> None:
        self._scene._ensure_open()
        self._check_member(light)
        self._scene.objects.remove(light)

    def clear(self) -> None:
        self._scene._ensure_open()
        for light in list(self._items.values()):
            self._scene.objects.remove(light)


class Scene:
    """A renderer scene with Blender-style data-block collections.

    Parameters
    ----------
    width, height:
        Render resolution in pixels.
    samples:
        Samples per pixel. Can be changed later through :attr:`samples`.
    seed:
        Initial integrator seed. Can be changed through :attr:`seed` or
        ``render(seed=...)``.
    max_bounces:
        Maximum path depth. Fixed for the lifetime of the scene.
    denoise, adaptive_sampling:
        Renderer options fixed for the lifetime of the scene.
    device:
        ``"AUTO"`` or ``"OPTIX"``. Only OptiX is currently available.
    history_samples:
        Experimental GPU reprojection of up to 0..64 previous raw frames before
        denoising. Default 0 preserves independent renders. Does not change samples.
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
        history_samples: int = 0,
    ) -> None:
        self._session = Session(
            width=width,
            height=height,
            samples=samples,
            seed=seed,
            max_bounces=max_bounces,
            denoise=denoise,
            adaptive_sampling=adaptive_sampling,
            device=device,
            denoiser=denoiser, denoise_passes=denoise_passes,
            denoise_prefilter=denoise_prefilter, denoise_quality=denoise_quality,
            denoise_use_gpu=denoise_use_gpu, denoise_upscale_quality=denoise_upscale_quality,
            viewport=viewport, render_mode=render_mode,
            samples_per_frame=samples_per_frame,
            history_samples=history_samples,
        )
        self._closed = False
        self._samples = int(samples)
        self._seed = int(seed)
        self.materials = _MaterialCollection(self)
        self.meshes = _MeshCollection(self)
        self.objects = _ObjectCollection(self)
        self.lights = _LightCollection(self)
        self.world = World(self)
        self._camera = Camera()
        self._camera._bind(self)

    @property
    def _session_obj(self) -> Session:
        return self._session

    def _ensure_open(self) -> None:
        if self._closed:
            raise PhotaraClosedError("Scene is closed")

    def _resolve_material(self, material: Optional[Material]) -> Optional[Material]:
        if material is None:
            return None
        if not isinstance(material, Material):
            raise TypeError("material must be a photara_rs.Material")
        if material.bound:
            if material._session is not self._session:
                raise ValueError("Material belongs to another Scene")
            if self.materials.get(material.name) is not material:
                material.name = self.materials._unique_name(material.name, "Material")
                self.materials._add(material)
            return material
        material.name = self.materials._unique_name(material.name, "Material")
        self._session.add_material(material)
        self.materials._add(material)
        return material

    # ------------------------------------------------------------------
    # Rendering.
    # ------------------------------------------------------------------
    @property
    def samples(self) -> int:
        return self._samples

    @samples.setter
    def samples(self, value: int) -> None:
        self._ensure_open()
        self._session.set_samples(value)
        self._samples = int(value)

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._ensure_open()
        self._session.set_seed(value)
        self._seed = int(value)

    @property
    def width(self) -> int:
        return self._session.width

    @property
    def height(self) -> int:
        return self._session.height

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    @property
    def device_name(self) -> str:
        return self._session.device_name

    @property
    def max_bounces(self) -> int:
        return self._session.max_bounces

    @property
    def denoise(self) -> bool:
        return self._session.denoise

    @property
    def adaptive_sampling(self) -> bool:
        return self._session.adaptive_sampling

    def omm_stats(self):
        """Idle-context hardware capability and attached opacity micromaps."""
        self._ensure_open()
        return self._session.omm_stats()

    def neeat_stats(self):
        """Return idle-context light count, global PDF and screen-tile diagnostics."""
        return self._session.neeat_stats()

    @property
    def texture_lod(self):
        """Active texture footprint estimator for both render modes."""
        self._ensure_open()
        return self._session.texture_lod

    def configure_texture_lod(self, method="RAY_CONES"):
        """Select RAY_CONES (default) or legacy compact RAY_DIFFERENTIALS."""
        self._ensure_open()
        self._session.configure_texture_lod(method)

    def configure_neeat(self, enabled=True):
        """Enable the experimental RealTime global and screen-tile light proposal."""
        self._session.configure_neeat(enabled)

    def configure_nee_candidates(self, count=1):
        """Set 1–16 surface NEE candidates; Reference retains one candidate."""
        self._ensure_open()
        self._session.configure_nee_candidates(count)

    @property
    def history_samples(self):
        """Previous raw frames available to pre-denoise reprojection; zero disables it."""
        self._ensure_open()
        return self._session.history_samples

    @history_samples.setter
    def history_samples(self, value):
        self._ensure_open()
        self._session.history_samples = value

    def reset_history(self):
        self._ensure_open()
        self._session.reset_history()

    def history_stats(self):
        self._ensure_open()
        return self._session.history_stats()

    def configure_render_mode(self, mode="REFERENCE", *, samples_per_frame=1):
        """Select Reference or the experimental frame-scheduled RealTime baseline."""
        self._ensure_open()
        self._session.configure_render_mode(mode, samples_per_frame=samples_per_frame)

    def render(self, seed: Optional[int] = None) -> RenderResult:
        """Render synchronously and capture all passes."""
        self._ensure_open()
        if seed is not None:
            seed = int(seed)
            if seed < 0:
                raise ValueError("seed must be non-negative")
            self._seed = seed
        self._session.render(seed=seed)
        return RenderResult(self._session, seed=self._seed)

    def resize(self, width: int, height: int) -> None:
        """Resize an idle scene's render buffers without rebuilding its data."""
        self._ensure_open()
        self._session.resize(width, height)

    def add_gaussian_splats(self, positions, scales, **settings):
        """Add a Gaussian cloud. Colors are encoded RGB, rotations are wxyz."""
        from .gsplat import GaussianSplats
        handle = self._session.add_gaussian_splats(positions, scales, **settings)
        return GaussianSplats(self, handle)

    def load_gaussian_splats(self, path, **settings):
        """Load a standard 3DGS PLY or .splat cloud into this scene."""
        from .gsplat import load_gaussians
        data = load_gaussians(path)
        return self.add_gaussian_splats(**data, **settings)

    def configure_sampling(self, **settings):
        """Configure sampling thresholds, minimum samples and time limit (seconds)."""
        self._session.configure_sampling(**settings)

    def configure_film(self, **settings):
        """Configure film exposure, transparency and pixel filtering."""
        self._ensure_open()
        self._session.configure_film(**settings)

    def render_async(self, seed: Optional[int] = None) -> RenderJob:
        """Start a non-blocking render and return a :class:`RenderJob`."""
        self._ensure_open()
        job = self._session.render_async(seed=seed)
        self._seed = self._session.seed
        return job

    # ------------------------------------------------------------------
    # Camera / world / lifecycle.
    # ------------------------------------------------------------------
    @property
    def camera(self) -> Camera:
        return self._camera

    @camera.setter
    def camera(self, value: Camera) -> None:
        self._ensure_open()
        if not isinstance(value, Camera):
            raise TypeError("camera must be a photara_rs.Camera")
        if value is not self._camera:
            value._bind(self)
            self._camera._scene = None
            self._camera = value

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release the GPU context. Scene items become unusable."""
        if not self._closed:
            self._session.close()
            self._closed = True

    def clear_objects(self) -> None:
        self.objects.clear()

    def clear_lights(self) -> None:
        self.lights.clear()

    def import_obj(self, filepath, **kwargs):
        """Import an OBJ/MTL file into this scene; see ``importers.load_obj``."""
        self._ensure_open()
        from .importers.obj import load_obj

        return load_obj(self, filepath, **kwargs)

    def import_gltf(self, filepath, **kwargs):
        """Import a glTF/GLB file into this scene; see ``importers.load_gltf``."""
        self._ensure_open()
        from .importers.gltf import load_gltf

        return load_gltf(self, filepath, **kwargs)

    def __enter__(self) -> "Scene":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __repr__(self) -> str:
        if self._closed:
            return "Scene(closed)"
        return (
            f"Scene({self.width}x{self.height}, samples={self.samples}, "
            f"objects={len(self.objects)}, device={self.device_name!r})"
        )


__all__ = [
    "Camera",
    "Light",
    "Mesh",
    "Object",
    "RenderResult",
    "Scene",
    "World",
    "linear_to_srgb",
    "save_png",
]
