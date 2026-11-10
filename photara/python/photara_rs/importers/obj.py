"""Pure-Python Wavefront OBJ/MTL importer."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._common import ImportResult, compute_vertex_normals, import_transform, srgb_to_linear
from ._errors import ObjImportError


@dataclass
class ObjMaterial:
    name: str
    base_color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    roughness: float = 0.5
    emission_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    emission_strength: float = 0.0
    base_color_texture: Optional[str] = None


@dataclass
class ObjPart:
    name: str
    material_name: Optional[str]
    vertices: np.ndarray
    indices: np.ndarray
    normals: Optional[np.ndarray]
    uvs: Optional[np.ndarray]


@dataclass
class ObjModel:
    parts: list[ObjPart]
    materials: dict[str, ObjMaterial]
    warnings: list[str] = field(default_factory=list)
    source: Optional[str] = None


def _logical_lines(text: str):
    """Yield (line_number, logical_line) while joining backslash continuations."""
    buffer = ""
    start_line = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not buffer:
            start_line = line_number
        if line.rstrip().endswith("\\"):
            buffer += line.rstrip()[:-1] + " "
            continue
        yield start_line, buffer + line
        buffer = ""
    if buffer:
        yield start_line, buffer


def _resolve_index(raw: str, count: int, kind: str, path: str, line: int) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ObjImportError(f"invalid {kind} index {raw!r}", path=path, line=line) from exc
    if value == 0:
        raise ObjImportError(f"{kind} index 0 is invalid", path=path, line=line)
    index = value - 1 if value > 0 else count + value
    if index < 0 or index >= count:
        raise ObjImportError(
            f"{kind} index {value} is out of range (count {count})", path=path, line=line
        )
    return index


def _path_from_token(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


def _resolve_relative(base_dir: str, value: str) -> str:
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(base_dir, value))


def _read_text(path: str, error_type=ObjImportError) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError as exc:
        raise error_type(f"cannot read file: {exc}", path=path) from exc


def _parse_mtl(path: str, warnings: list[str], strict: bool) -> dict[str, ObjMaterial]:
    base_dir = os.path.dirname(os.path.abspath(path))
    text = _read_text(path)
    materials: dict[str, ObjMaterial] = {}
    current: Optional[ObjMaterial] = None
    current_name: Optional[str] = None

    for line_number, logical in _logical_lines(text):
        line = logical.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        key = tokens[0].lower()
        args = tokens[1:]

        if key == "newmtl":
            current_name = " ".join(args) if args else f"Material{len(materials) + 1}"
            current = ObjMaterial(name=current_name)
            materials[current_name] = current
            continue

        if current is None:
            if strict:
                raise ObjImportError(
                    "material property before newmtl", path=path, line=line_number
                )
            continue

        try:
            if key == "kd":
                values = tuple(float(value) for value in args[:3])
                if len(values) != 3:
                    raise ValueError("expected 3 values")
                linear = srgb_to_linear(values)
                current.base_color = tuple(float(value) for value in linear)
            elif key == "ke":
                values = tuple(float(value) for value in args[:3])
                if len(values) != 3:
                    raise ValueError("expected 3 values")
                linear = srgb_to_linear(values)
                current.emission_color = tuple(float(value) for value in linear)
                current.emission_strength = 1.0 if float(max(values)) > 0.0 else 0.0
            elif key == "ns":
                exponent = float(args[0])
                current.roughness = float(
                    np.clip(math.sqrt(2.0 / (max(exponent, 0.0) + 2.0)), 0.0, 1.0)
                )
            elif key in ("d", "tr"):
                alpha = float(args[0])
                if key == "tr":
                    alpha = 1.0 - alpha
                if alpha < 1.0:
                    warnings.append(
                        f"{path}:{line_number}: material '{current.name}' alpha is not "
                        "represented by the renderer; alpha ignored"
                    )
            elif key == "map_kd":
                if not args:
                    raise ValueError("expected a texture path")
                texture = _path_from_token(args[-1])
                current.base_color_texture = _resolve_relative(base_dir, texture)
                if not os.path.isfile(current.base_color_texture):
                    message = (
                        f"{path}:{line_number}: missing base-color texture "
                        f"{current.base_color_texture}"
                    )
                    if strict:
                        raise ObjImportError(message)
                    warnings.append(message)
                    current.base_color_texture = None
            elif key in ("map_ks", "map_ns", "map_bump", "bump", "disp", "decal", "norm"):
                warnings.append(
                    f"{path}:{line_number}: unsupported material map '{key}' ignored"
                )
        except (IndexError, ValueError) as exc:
            raise ObjImportError(
                f"invalid {key} value: {exc}", path=path, line=line_number
            ) from exc

    return materials


def parse_obj(filepath, *, strict: bool = False) -> ObjModel:
    """Parse an OBJ and its MTL files without creating renderer objects."""
    path = os.path.abspath(os.fspath(filepath))
    if not os.path.isfile(path):
        raise ObjImportError("file does not exist", path=path)

    text = _read_text(path)
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[tuple[str, Optional[str], list[tuple[int, Optional[int], Optional[int]]], int]] = []
    mtllibs: list[str] = []
    warnings: list[str] = []

    current_object = "default"
    current_group: Optional[str] = None
    current_material: Optional[str] = None

    for line_number, logical in _logical_lines(text):
        line = logical.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        key = tokens[0].lower()
        args = tokens[1:]

        try:
            if key == "v":
                if len(args) < 3:
                    raise ValueError("expected at least 3 coordinates")
                x, y, z = (float(value) for value in args[:3])
                positions.append((x, y, z))
            elif key == "vn":
                if len(args) < 3:
                    raise ValueError("expected 3 components")
                normals.append(tuple(float(value) for value in args[:3]))
            elif key == "vt":
                if len(args) < 2:
                    raise ValueError("expected at least 2 components")
                uvs.append((float(args[0]), float(args[1])))
            elif key == "f":
                if len(args) < 3:
                    raise ValueError("expected at least 3 corners")
                corners: list[tuple[int, Optional[int], Optional[int]]] = []
                for token in args:
                    parts = token.split("/")
                    if len(parts) > 3 or not parts[0]:
                        raise ValueError(f"invalid face corner {token!r}")
                    vertex = _resolve_index(parts[0], len(positions), "vertex", path, line_number)
                    uv = None
                    normal = None
                    if len(parts) >= 2 and parts[1]:
                        uv = _resolve_index(parts[1], len(uvs), "UV", path, line_number)
                    if len(parts) >= 3 and parts[2]:
                        normal = _resolve_index(
                            parts[2], len(normals), "normal", path, line_number
                        )
                    corners.append((vertex, uv, normal))

                group = current_group or current_object
                for index in range(1, len(corners) - 1):
                    triangle = [corners[0], corners[index], corners[index + 1]]
                    if len({corner[0] for corner in triangle}) < 3:
                        continue
                    faces.append((group, current_material, triangle, line_number))
            elif key == "o":
                current_object = " ".join(args) if args else "default"
                current_group = None
            elif key == "g":
                current_group = " ".join(args) if args else None
            elif key == "usemtl":
                current_material = " ".join(args) if args else None
            elif key == "mtllib":
                mtllibs.extend(_path_from_token(value) for value in args)
            elif key in ("s", "l", "p"):
                # Smoothing groups, lines and points do not map to the mesh API.
                continue
        except (IndexError, ValueError) as exc:
            raise ObjImportError(str(exc), path=path, line=line_number) from exc

    if not faces:
        raise ObjImportError("no triangles found", path=path)

    materials: dict[str, ObjMaterial] = {}
    base_dir = os.path.dirname(path)
    for mtllib in mtllibs:
        mtl_path = _resolve_relative(base_dir, mtllib)
        if not os.path.isfile(mtl_path):
            message = f"{path}: missing material library {mtl_path}"
            if strict:
                raise ObjImportError(message)
            warnings.append(message)
            continue
        materials.update(_parse_mtl(mtl_path, warnings, strict))

    grouped: dict[tuple[str, Optional[str]], list] = {}
    for group, material, triangle, _line in faces:
        grouped.setdefault((group, material), []).append(triangle)

    parts: list[ObjPart] = []
    skipped_degenerate = 0
    for (group, material_name), triangles in grouped.items():
        all_uv = all(
            uv is not None for triangle in triangles for _v, uv, _n in triangle
        )
        all_normal = all(
            normal is not None for triangle in triangles for _v, _uv, normal in triangle
        )
        any_uv = any(
            uv is not None for triangle in triangles for _v, uv, _n in triangle
        )
        any_normal = any(
            normal is not None for triangle in triangles for _v, _uv, normal in triangle
        )
        if any_uv and not all_uv:
            warnings.append(
                f"{path}: group '{group}' has partial UVs; UVs omitted for this part"
            )
        if any_normal and not all_normal:
            warnings.append(
                f"{path}: group '{group}' has partial normals; normals recomputed for this part"
            )

        remap: dict[tuple, int] = {}
        vertices: list[tuple[float, float, float]] = []
        uv_values: list[tuple[float, float]] = []
        normal_values: list[tuple[float, float, float]] = []
        indices: list[int] = []

        for triangle in triangles:
            for vertex_index, uv_index, normal_index in triangle:
                key = (
                    vertex_index,
                    uv_index if all_uv else None,
                    normal_index if all_normal else None,
                )
                mapped = remap.get(key)
                if mapped is None:
                    mapped = len(vertices)
                    remap[key] = mapped
                    vertices.append(positions[vertex_index])
                    if all_uv:
                        uv_values.append(uvs[uv_index])  # type: ignore[index]
                    if all_normal:
                        normal_values.append(normals[normal_index])  # type: ignore[index]
                indices.append(mapped)

        vertices_array = np.asarray(vertices, dtype=np.float32)
        indices_array = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
        if indices_array.shape[0] == 0:
            skipped_degenerate += 1
            continue

        if all_normal:
            normals_array = np.asarray(normal_values, dtype=np.float32)
            lengths = np.linalg.norm(normals_array, axis=1)
            valid = lengths > 1e-12
            normals_array[valid] /= lengths[valid, np.newaxis]
            normals_array[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            normals_array = compute_vertex_normals(vertices_array, indices_array)

        uvs_array = (
            np.asarray(uv_values, dtype=np.float32) if all_uv else None
        )
        parts.append(
            ObjPart(
                name=group,
                material_name=material_name,
                vertices=vertices_array,
                indices=indices_array,
                normals=normals_array,
                uvs=uvs_array,
            )
        )

    if skipped_degenerate:
        warnings.append(f"{path}: skipped {skipped_degenerate} empty part(s)")
    if not parts:
        raise ObjImportError("no non-degenerate triangles found", path=path)

    return ObjModel(parts=parts, materials=materials, warnings=warnings, source=path)


def load_obj(
    scene,
    filepath,
    *,
    name: Optional[str] = None,
    location=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0),
    scale=1.0,
    strict: bool = False,
    object_id_start: int = 0,
) -> ImportResult:
    """Parse an OBJ and create meshes, objects and materials in a Scene."""
    model = parse_obj(filepath, strict=strict)
    warnings = list(model.warnings)
    transform = import_transform(location, rotation, scale)
    stem = os.path.splitext(os.path.basename(model.source or "object"))[0]

    material_cache: dict[str, object] = {}
    created_materials: list = []
    fallback_material = None

    def make_fallback():
        nonlocal fallback_material
        if fallback_material is None:
            fallback_material = scene.materials.new(
                f"{stem} default", base_color=(0.8, 0.8, 0.8), roughness=0.5
            )
            created_materials.append(fallback_material)
        return fallback_material

    def make_material(obj_material: ObjMaterial):
        if obj_material.name not in material_cache:
            material = scene.materials.new(
                obj_material.name,
                base_color=obj_material.base_color,
                roughness=obj_material.roughness,
                emission_color=obj_material.emission_color,
                emission_strength=obj_material.emission_strength,
            )
            if obj_material.base_color_texture is not None:
                try:
                    material.base_color_texture = {
                        "path": obj_material.base_color_texture,
                        "colorspace": "sRGB",
                    }
                except (FileNotFoundError, ValueError) as exc:
                    message = (
                        f"material '{obj_material.name}': cannot apply texture "
                        f"{obj_material.base_color_texture}: {exc}"
                    )
                    if strict:
                        raise ObjImportError(message, path=model.source)
                    warnings.append(message)
            material_cache[obj_material.name] = material
            created_materials.append(material)
        return material_cache[obj_material.name]

    objects = []
    meshes = []
    multi_part = len(model.parts) > 1
    for index, part in enumerate(model.parts):
        material = None
        if part.material_name:
            obj_material = model.materials.get(part.material_name)
            if obj_material is None:
                message = (
                    f"{model.source}: material '{part.material_name}' not found; "
                    "using fallback material"
                )
                if strict:
                    raise ObjImportError(message)
                warnings.append(message)
                material = make_fallback()
            else:
                material = make_material(obj_material)
        else:
            material = make_fallback()

        part_name = part.name or "object"
        if name is not None:
            object_name = name if not multi_part else f"{name}_{part_name}"
        else:
            object_name = part_name
        object_id = 0 if object_id_start == 0 else object_id_start + index
        mesh = scene.meshes.new(
            object_name,
            part.vertices,
            part.indices,
            normals=part.normals,
            uvs=part.uvs,
            material=material,
        )
        obj = scene.objects.new(
            object_name, mesh, transform=transform, object_id=object_id
        )
        meshes.append(mesh)
        objects.append(obj)

    return ImportResult(
        objects=objects,
        meshes=meshes,
        materials=created_materials,
        warnings=warnings,
        source=model.source,
    )


__all__ = [
    "ObjMaterial",
    "ObjModel",
    "ObjPart",
    "load_obj",
    "parse_obj",
]
