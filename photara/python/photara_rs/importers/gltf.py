"""Pure-Python glTF 2.0 / GLB importer."""

from __future__ import annotations

import base64
import json
import os
import struct
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._common import ImportResult, compute_vertex_normals, import_transform
from ._errors import GltfImportError

_GLB_MAGIC = 0x46546C67
_GLB_JSON = 0x4E4F534A
_GLB_BIN = 0x004E4942

_COMPONENT_TYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/vnd-ms.dds": ".dds",
    "image/x-exr": ".exr",
    "image/hdr": ".hdr",
}

_WRAP_TO_EXTENSION = {
    10497: "repeat",
    33071: "extend",
    33648: "mirror",
}

_FILTER_TO_INTERPOLATION = {
    9728: "closest",
    9729: "linear",
    9984: "linear",
    9985: "linear",
    9986: "linear",
    9987: "linear",
}


@dataclass
class GltfPrimitive:
    name: str
    material_index: Optional[int]
    vertices: np.ndarray
    indices: np.ndarray
    normals: Optional[np.ndarray]
    uvs: Optional[np.ndarray]


@dataclass
class GltfNode:
    name: str
    matrix: np.ndarray
    primitives: list[GltfPrimitive]


@dataclass
class GltfMaterialData:
    name: str
    base_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    metallic: float = 1.0
    roughness: float = 1.0
    emission_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    emission_strength: float = 0.0
    base_color_texture: Optional[str] = None
    texture_interpolation: str = "linear"
    texture_extension: str = "repeat"


@dataclass
class GltfDocument:
    nodes: list[GltfNode]
    materials: list[GltfMaterialData]
    warnings: list[str] = field(default_factory=list)
    source: Optional[str] = None
    texture_cache_dir: Optional[str] = None


def _read_json_file(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except OSError as exc:
        raise GltfImportError(f"cannot read file: {exc}", path=path) from exc
    except json.JSONDecodeError as exc:
        raise GltfImportError(
            f"invalid JSON: {exc.msg}", path=path, line=exc.lineno
        ) from exc


def _read_glb(path: str):
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise GltfImportError(f"cannot read file: {exc}", path=path) from exc

    if len(data) < 12:
        raise GltfImportError("GLB header is truncated", path=path)
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC or version != 2:
        raise GltfImportError("not a glTF 2.0 GLB file", path=path)
    if length > len(data) or length < 20:
        raise GltfImportError("GLB length is invalid", path=path)

    offset = 12
    json_chunk = None
    bin_chunk = None
    while offset + 8 <= length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        if offset + chunk_length > length:
            raise GltfImportError("GLB chunk is truncated", path=path)
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == _GLB_JSON:
            json_chunk = chunk
        elif chunk_type == _GLB_BIN:
            bin_chunk = chunk

    if json_chunk is None:
        raise GltfImportError("GLB has no JSON chunk", path=path)
    try:
        document = json.loads(json_chunk.decode("utf-8-sig").rstrip("\x00 "))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GltfImportError(f"invalid GLB JSON: {exc}", path=path) from exc
    return document, bin_chunk


def _decode_data_uri(uri: str) -> tuple[bytes, str]:
    if not uri.startswith("data:"):
        raise ValueError("not a data URI")
    header, _, payload = uri.partition(",")
    metadata = header[5:]
    mime = metadata.split(";", 1)[0] or "application/octet-stream"
    if ";base64" in metadata:
        data = base64.b64decode(payload)
    else:
        data = urllib.parse.unquote_to_bytes(payload)
    return data, mime


def _resolve_asset_path(base_dir: str, uri: str) -> str:
    uri = urllib.parse.unquote(uri)
    if os.path.isabs(uri):
        return os.path.abspath(uri)
    return os.path.abspath(os.path.join(base_dir, uri))


def _load_buffers(document: dict, base_dir: str, bin_chunk, path: str, strict: bool, warnings: list[str]):
    buffers = []
    for index, buffer in enumerate(document.get("buffers", [])):
        uri = buffer.get("uri")
        if uri is None:
            if bin_chunk is None:
                raise GltfImportError(f"buffer {index} has no URI and no GLB BIN chunk", path=path)
            data = bin_chunk
        elif uri.startswith("data:"):
            data, _mime = _decode_data_uri(uri)
        else:
            filename = _resolve_asset_path(base_dir, uri)
            if not os.path.isfile(filename):
                message = f"{path}: missing buffer {filename}"
                if strict:
                    raise GltfImportError(message, path=path)
                warnings.append(message)
                data = b""
            else:
                with open(filename, "rb") as handle:
                    data = handle.read()
        expected = buffer.get("byteLength")
        if expected is not None and len(data) < int(expected):
            message = f"{path}: buffer {index} is shorter than byteLength"
            if strict:
                raise GltfImportError(message, path=path)
            warnings.append(message)
        buffers.append(memoryview(data))
    return buffers


def _read_accessor(document: dict, buffers: list, accessor_index: int, path: str) -> np.ndarray:
    accessors = document.get("accessors", [])
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise GltfImportError(f"accessor {accessor_index} is out of range", path=path)
    accessor = accessors[accessor_index]
    if accessor.get("sparse") is not None:
        raise GltfImportError(f"accessor {accessor_index} uses unsupported sparse data", path=path)

    component_type = accessor.get("componentType")
    dtype = _COMPONENT_TYPES.get(component_type)
    if dtype is None:
        raise GltfImportError(
            f"accessor {accessor_index} has unsupported componentType {component_type}", path=path
        )
    type_name = accessor.get("type")
    components = _TYPE_COMPONENTS.get(type_name)
    if components is None:
        raise GltfImportError(
            f"accessor {accessor_index} has unsupported type {type_name}", path=path
        )
    count = int(accessor.get("count", 0))
    if count <= 0:
        raise GltfImportError(f"accessor {accessor_index} has no elements", path=path)

    view_index = accessor.get("bufferView")
    if view_index is None:
        raise GltfImportError(f"accessor {accessor_index} has no bufferView", path=path)
    views = document.get("bufferViews", [])
    if view_index < 0 or view_index >= len(views):
        raise GltfImportError(f"bufferView {view_index} is out of range", path=path)
    view = views[view_index]
    buffer_index = int(view.get("buffer", 0))
    if buffer_index < 0 or buffer_index >= len(buffers):
        raise GltfImportError(f"buffer {buffer_index} is out of range", path=path)
    buffer = buffers[buffer_index]

    item_size = np.dtype(dtype).itemsize
    element_size = item_size * components
    stride = int(view.get("byteStride", element_size))
    if stride < element_size:
        raise GltfImportError(
            f"bufferView {view_index} byteStride is smaller than its element size", path=path
        )
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    required = offset + stride * (count - 1) + element_size
    if required > len(buffer):
        raise GltfImportError(f"accessor {accessor_index} reads past its buffer", path=path)

    array = np.ndarray(
        shape=(count, components),
        dtype=np.dtype(dtype),
        buffer=buffer,
        offset=offset,
        strides=(stride, item_size),
    ).copy()

    normalized = bool(accessor.get("normalized", False))
    if normalized and dtype in (np.int8, np.uint8, np.int16, np.uint16):
        info = np.iinfo(dtype)
        if info.min < 0:
            array = np.maximum(array.astype(np.float32) / abs(info.min), -1.0)
        else:
            array = array.astype(np.float32) / info.max
    if components == 1:
        return array.reshape(-1)
    return array


def _quaternion_matrix(quaternion) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if values.shape != (4,):
        raise ValueError("rotation must contain 4 values")
    x, y, z, w = (float(value) for value in values)
    length = (x * x + y * y + z * z + w * w) ** 0.5
    if length <= 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / length, y / length, z / length, w / length
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _node_local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        values = np.asarray(node["matrix"], dtype=np.float32).reshape(-1)
        if values.shape != (16,):
            raise ValueError("node matrix must contain 16 values")
        # glTF matrices are column-major; Photara uses row-major rows.
        return values.reshape(4, 4).T.copy()

    matrix = np.eye(4, dtype=np.float32)
    translation = np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float32)
    scale = np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float32)
    rotation = _quaternion_matrix(node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    matrix[:3, :3] = rotation * scale[np.newaxis, :]
    matrix[:3, 3] = translation
    return matrix


def _sampler_to_texture_options(sampler: dict) -> tuple[str, str]:
    wrap_s = int(sampler.get("wrapS", 10497))
    wrap_t = int(sampler.get("wrapT", 10497))
    extension = _WRAP_TO_EXTENSION.get(wrap_s, "repeat")
    if wrap_s != wrap_t:
        extension = _WRAP_TO_EXTENSION.get(wrap_s, "repeat")
    mag = sampler.get("magFilter")
    min_filter = sampler.get("minFilter")
    if mag is not None:
        interpolation = _FILTER_TO_INTERPOLATION.get(int(mag), "linear")
    elif min_filter is not None:
        interpolation = _FILTER_TO_INTERPOLATION.get(int(min_filter), "linear")
    else:
        interpolation = "linear"
    return interpolation, extension


def _load_image(
    document: dict,
    buffers: list,
    base_dir: str,
    image_index: int,
    cache: dict,
    temp_dir: list,
    path: str,
    strict: bool,
    warnings: list[str],
    extract_embedded_textures: bool,
) -> Optional[str]:
    if image_index in cache:
        return cache[image_index]
    images = document.get("images", [])
    if image_index < 0 or image_index >= len(images):
        raise GltfImportError(f"image {image_index} is out of range", path=path)
    image = images[image_index]
    filename = None

    if "uri" in image:
        uri = image["uri"]
        if uri.startswith("data:"):
            if not extract_embedded_textures:
                warnings.append(f"{path}: embedded image {image_index} skipped")
                cache[image_index] = None
                return None
            try:
                data, mime = _decode_data_uri(uri)
            except ValueError as exc:
                raise GltfImportError(f"invalid image data URI: {exc}", path=path) from exc
            if temp_dir[0] is None:
                temp_dir[0] = tempfile.mkdtemp(prefix="photara_gltf_")
            extension = _MIME_EXTENSIONS.get(mime, ".bin")
            filename = os.path.join(temp_dir[0], f"image_{image_index}{extension}")
            with open(filename, "wb") as handle:
                handle.write(data)
        else:
            filename = _resolve_asset_path(base_dir, uri)
            if not os.path.isfile(filename):
                message = f"{path}: missing image {filename}"
                if strict:
                    raise GltfImportError(message, path=path)
                warnings.append(message)
                cache[image_index] = None
                return None
    elif "bufferView" in image:
        if not extract_embedded_textures:
            warnings.append(f"{path}: embedded image {image_index} skipped")
            cache[image_index] = None
            return None
        view_index = int(image["bufferView"])
        views = document.get("bufferViews", [])
        if view_index < 0 or view_index >= len(views):
            raise GltfImportError(f"image bufferView {view_index} is out of range", path=path)
        view = views[view_index]
        buffer_index = int(view.get("buffer", 0))
        if buffer_index < 0 or buffer_index >= len(buffers):
            raise GltfImportError(f"image buffer {buffer_index} is out of range", path=path)
        start = int(view.get("byteOffset", 0))
        length = int(view.get("byteLength", 0))
        data = bytes(buffers[buffer_index][start : start + length])
        mime = image.get("mimeType", "application/octet-stream")
        if mime not in _MIME_EXTENSIONS:
            warnings.append(f"{path}: image {image_index} uses unsupported MIME {mime}")
        if temp_dir[0] is None:
            temp_dir[0] = tempfile.mkdtemp(prefix="photara_gltf_")
        extension = _MIME_EXTENSIONS.get(mime, ".bin")
        filename = os.path.join(temp_dir[0], f"image_{image_index}{extension}")
        with open(filename, "wb") as handle:
            handle.write(data)
    else:
        message = f"{path}: image {image_index} has neither URI nor bufferView"
        if strict:
            raise GltfImportError(message, path=path)
        warnings.append(message)
        cache[image_index] = None
        return None

    cache[image_index] = filename
    return filename


def _texture_options(
    document: dict,
    buffers: list,
    base_dir: str,
    texture_index: int,
    image_cache: dict,
    temp_dir: list,
    path: str,
    strict: bool,
    warnings: list[str],
    extract_embedded_textures: bool,
):
    textures = document.get("textures", [])
    if texture_index < 0 or texture_index >= len(textures):
        raise GltfImportError(f"texture {texture_index} is out of range", path=path)
    texture = textures[texture_index]
    source = texture.get("source")
    if source is None:
        message = f"{path}: texture {texture_index} has no image source"
        if strict:
            raise GltfImportError(message, path=path)
        warnings.append(message)
        return None
    image_path = _load_image(
        document,
        buffers,
        base_dir,
        int(source),
        image_cache,
        temp_dir,
        path,
        strict,
        warnings,
        extract_embedded_textures,
    )
    if image_path is None:
        return None
    sampler_index = texture.get("sampler", 0)
    samplers = document.get("samplers", [])
    sampler = samplers[sampler_index] if 0 <= sampler_index < len(samplers) else {}
    interpolation, extension = _sampler_to_texture_options(sampler)
    return image_path, interpolation, extension


def _parse_material(
    document: dict,
    buffers: list,
    base_dir: str,
    material_index: int,
    image_cache: dict,
    temp_dir: list,
    path: str,
    strict: bool,
    warnings: list[str],
    extract_embedded_textures: bool,
) -> GltfMaterialData:
    materials = document.get("materials", [])
    material = materials[material_index] if 0 <= material_index < len(materials) else {}
    pbr = material.get("pbrMetallicRoughness", {})
    base_color = np.asarray(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0)), dtype=np.float32)
    if base_color.shape != (4,):
        raise GltfImportError(f"material {material_index} has invalid baseColorFactor", path=path)
    if float(base_color[3]) < 1.0:
        warnings.append(
            f"{path}: material {material_index} baseColor alpha is not represented; alpha ignored"
        )
    metallic = float(pbr.get("metallicFactor", 1.0))
    roughness = float(pbr.get("roughnessFactor", 1.0))
    emissive = np.asarray(material.get("emissiveFactor", (0.0, 0.0, 0.0)), dtype=np.float32)
    if emissive.shape != (3,):
        raise GltfImportError(f"material {material_index} has invalid emissiveFactor", path=path)

    texture_path = None
    interpolation = "linear"
    extension = "repeat"
    texture_info = pbr.get("baseColorTexture")
    if texture_info:
        if int(texture_info.get("texCoord", 0)) != 0:
            warnings.append(
                f"{path}: material {material_index} uses TEXCOORD_1 or higher; UV0 is used"
            )
        options = _texture_options(
            document,
            buffers,
            base_dir,
            int(texture_info.get("index", 0)),
            image_cache,
            temp_dir,
            path,
            strict,
            warnings,
            extract_embedded_textures,
        )
        if options is not None:
            texture_path, interpolation, extension = options

    alpha_mode = str(material.get("alphaMode", "OPAQUE")).upper()
    if alpha_mode != "OPAQUE":
        warnings.append(
            f"{path}: material {material_index} alphaMode={alpha_mode} is not supported; treated as opaque"
        )
    if material.get("doubleSided", False):
        warnings.append(
            f"{path}: material {material_index} doubleSided is not represented by the renderer"
        )

    return GltfMaterialData(
        name=str(material.get("name") or f"Material_{material_index}"),
        base_color=tuple(float(value) for value in base_color[:3]),
        metallic=float(np.clip(metallic, 0.0, 1.0)),
        roughness=float(np.clip(roughness, 0.0, 1.0)),
        emission_color=tuple(float(value) for value in emissive),
        emission_strength=1.0 if float(np.max(emissive)) > 0.0 else 0.0,
        base_color_texture=texture_path,
        texture_interpolation=interpolation,
        texture_extension=extension,
    )


def _parse_primitive(
    document: dict,
    buffers: list,
    mesh_index: int,
    primitive_index: int,
    mesh_name: str,
    path: str,
    strict: bool,
    warnings: list[str],
) -> Optional[GltfPrimitive]:
    mesh = document["meshes"][mesh_index]
    primitives = mesh.get("primitives", [])
    if primitive_index < 0 or primitive_index >= len(primitives):
        raise GltfImportError(f"mesh {mesh_index} primitive is out of range", path=path)
    primitive = primitives[primitive_index]
    mode = int(primitive.get("mode", 4))
    if mode != 4:
        message = (
            f"{path}: mesh {mesh_index} primitive {primitive_index} mode {mode} "
            "is not TRIANGLES; skipped"
        )
        if strict:
            raise GltfImportError(message, path=path)
        warnings.append(message)
        return None

    attributes = primitive.get("attributes", {})
    if "POSITION" not in attributes:
        raise GltfImportError(f"mesh {mesh_index} primitive has no POSITION", path=path)
    vertices = _read_accessor(document, buffers, int(attributes["POSITION"]), path)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise GltfImportError(f"mesh {mesh_index} POSITION is not VEC3", path=path)
    vertices = vertices.astype(np.float32, copy=False)

    if "indices" in primitive:
        indices = _read_accessor(document, buffers, int(primitive["indices"]), path)
        indices = indices.astype(np.int64, copy=False).reshape(-1)
    else:
        indices = np.arange(vertices.shape[0], dtype=np.int64)
    if indices.size % 3 != 0:
        raise GltfImportError(f"mesh {mesh_index} indices are not triangles", path=path)
    if indices.size and (indices.min() < 0 or indices.max() >= vertices.shape[0]):
        raise GltfImportError(f"mesh {mesh_index} indices are out of range", path=path)
    indices = indices.astype(np.int32).reshape(-1, 3)

    normals = None
    if "NORMAL" in attributes:
        normals = _read_accessor(document, buffers, int(attributes["NORMAL"]), path)
        if normals.ndim != 2 or normals.shape != vertices.shape:
            raise GltfImportError(f"mesh {mesh_index} NORMAL is invalid", path=path)
        normals = normals.astype(np.float32, copy=False)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        normals[valid] /= lengths[valid, np.newaxis]
        normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        normals = compute_vertex_normals(vertices, indices)

    uvs = None
    if "TEXCOORD_0" in attributes:
        uvs = _read_accessor(document, buffers, int(attributes["TEXCOORD_0"]), path)
        if uvs.ndim != 2 or uvs.shape[0] != vertices.shape[0] or uvs.shape[1] < 2:
            raise GltfImportError(f"mesh {mesh_index} TEXCOORD_0 is invalid", path=path)
        uvs = uvs[:, :2].astype(np.float32, copy=True)
        # glTF uses a top-left UV origin; Photara uses the opposite V direction.
        uvs[:, 1] = 1.0 - uvs[:, 1]

    material_index = primitive.get("material")
    return GltfPrimitive(
        name=mesh_name,
        material_index=None if material_index is None else int(material_index),
        vertices=vertices.copy(),
        indices=indices.copy(),
        normals=normals.copy() if normals is not None else None,
        uvs=uvs,
    )


def parse_gltf(filepath, *, strict: bool = False, extract_embedded_textures: bool = True) -> GltfDocument:
    """Parse a glTF or GLB file into nodes, primitives and materials."""
    path = os.path.abspath(os.fspath(filepath))
    if not os.path.isfile(path):
        raise GltfImportError("file does not exist", path=path)

    if path.lower().endswith(".glb"):
        document, bin_chunk = _read_glb(path)
    else:
        document, bin_chunk = _read_json_file(path), None

    base_dir = os.path.dirname(path)
    warnings: list[str] = []
    buffers = _load_buffers(document, base_dir, bin_chunk, path, strict, warnings)

    required = document.get("extensionsRequired", [])
    supported_required = set()
    for extension in required:
        if extension not in supported_required:
            message = f"{path}: required glTF extension {extension} is not supported"
            if strict:
                raise GltfImportError(message, path=path)
            warnings.append(message)
    for key in ("skins", "animations", "cameras"):
        if document.get(key):
            warnings.append(f"{path}: glTF {key} are not imported")

    image_cache: dict[int, Optional[str]] = {}
    temp_dir = [None]
    materials = [
        _parse_material(
            document,
            buffers,
            base_dir,
            index,
            image_cache,
            temp_dir,
            path,
            strict,
            warnings,
            extract_embedded_textures,
        )
        for index in range(len(document.get("materials", [])))
    ]

    nodes_data = document.get("nodes", [])
    meshes_data = document.get("meshes", [])
    if not nodes_data:
        raise GltfImportError("glTF has no nodes", path=path)

    local_matrices = []
    for node in nodes_data:
        try:
            local_matrices.append(_node_local_matrix(node))
        except ValueError as exc:
            raise GltfImportError(str(exc), path=path) from exc

    child_set = set()
    for node in nodes_data:
        child_set.update(int(index) for index in node.get("children", []))
    scenes = document.get("scenes", [])
    scene_index = int(document.get("scene", 0))
    if 0 <= scene_index < len(scenes):
        roots = [int(index) for index in scenes[scene_index].get("nodes", [])]
    else:
        roots = [index for index in range(len(nodes_data)) if index not in child_set]
    if not roots:
        roots = [index for index, node in enumerate(nodes_data) if "mesh" in node]
    if not roots:
        raise GltfImportError("glTF scene has no root nodes", path=path)

    nodes: list[GltfNode] = []
    visited: set[int] = set()

    def visit(index: int, parent_matrix: np.ndarray, parent_name: str) -> None:
        if index < 0 or index >= len(nodes_data):
            raise GltfImportError(f"node {index} is out of range", path=path)
        if index in visited:
            warnings.append(f"{path}: node {index} is referenced more than once; duplicate skipped")
            return
        visited.add(index)
        node = nodes_data[index]
        world = parent_matrix @ local_matrices[index]
        node_name = str(node.get("name") or f"node_{index}")
        primitives: list[GltfPrimitive] = []
        if "mesh" in node:
            mesh_index = int(node["mesh"])
            if mesh_index < 0 or mesh_index >= len(meshes_data):
                raise GltfImportError(f"node {index} mesh is out of range", path=path)
            mesh = meshes_data[mesh_index]
            mesh_name = str(mesh.get("name") or f"mesh_{mesh_index}")
            for primitive_index in range(len(mesh.get("primitives", []))):
                primitive = _parse_primitive(
                    document,
                    buffers,
                    mesh_index,
                    primitive_index,
                    mesh_name,
                    path,
                    strict,
                    warnings,
                )
                if primitive is not None:
                    primitives.append(primitive)
        if primitives:
            nodes.append(GltfNode(name=node_name, matrix=world, primitives=primitives))
        for child in node.get("children", []):
            visit(int(child), world, node_name)

    identity = np.eye(4, dtype=np.float32)
    for root in roots:
        visit(root, identity, "")

    if not nodes:
        raise GltfImportError("glTF has no triangle primitives to import", path=path)

    return GltfDocument(
        nodes=nodes,
        materials=materials,
        warnings=warnings,
        source=path,
        texture_cache_dir=temp_dir[0],
    )


def load_gltf(
    scene,
    filepath,
    *,
    name: Optional[str] = None,
    location=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0),
    scale=1.0,
    strict: bool = False,
    extract_embedded_textures: bool = True,
    object_id_start: int = 0,
) -> ImportResult:
    """Parse a glTF/GLB file and create Scene materials, meshes and objects."""
    document = parse_gltf(
        filepath, strict=strict, extract_embedded_textures=extract_embedded_textures
    )
    warnings = list(document.warnings)
    outer = import_transform(location, rotation, scale)
    stem = os.path.splitext(os.path.basename(document.source or "asset"))[0]
    base_name = name or stem

    created_materials = []
    material_map = {}
    fallback_material = None

    def make_fallback():
        nonlocal fallback_material
        if fallback_material is None:
            fallback_material = scene.materials.new(
                f"{base_name} default", base_color=(0.8, 0.8, 0.8), roughness=0.5
            )
            created_materials.append(fallback_material)
        return fallback_material

    for index, data in enumerate(document.materials):
        material = scene.materials.new(
            data.name,
            base_color=data.base_color,
            metallic=data.metallic,
            roughness=data.roughness,
            emission_color=data.emission_color,
            emission_strength=data.emission_strength,
        )
        if data.base_color_texture is not None:
            try:
                material.base_color_texture = {
                    "path": data.base_color_texture,
                    "colorspace": "sRGB",
                    "interpolation": data.texture_interpolation,
                    "extension": data.texture_extension,
                }
            except (FileNotFoundError, ValueError) as exc:
                message = (
                    f"material '{data.name}': cannot apply texture "
                    f"{data.base_color_texture}: {exc}"
                )
                if strict:
                    raise GltfImportError(message, path=document.source) from exc
                warnings.append(message)
        material_map[index] = material
        created_materials.append(material)

    objects = []
    meshes = []
    for node_index, node in enumerate(document.nodes):
        node_name = node.name
        for primitive_index, primitive in enumerate(node.primitives):
            material = None
            if primitive.material_index is not None:
                material = material_map.get(primitive.material_index)
                if material is None:
                    message = (
                        f"{document.source}: material index {primitive.material_index} "
                        "is out of range; using fallback"
                    )
                    if strict:
                        raise GltfImportError(message, path=document.source)
                    warnings.append(message)
            if material is None:
                material = make_fallback()

            part_name = node_name
            if len(node.primitives) > 1:
                part_name = f"{node_name}_{primitive_index}"
            if name is not None:
                single_part = len(document.nodes) == 1 and len(node.primitives) == 1
                object_name = name if single_part else f"{name}_{part_name}"
            else:
                object_name = part_name
            mesh = scene.meshes.new(
                object_name,
                primitive.vertices,
                primitive.indices,
                normals=primitive.normals,
                uvs=primitive.uvs,
                material=material,
            )
            object_id = 0 if object_id_start == 0 else object_id_start + len(objects)
            obj = scene.objects.new(
                object_name,
                mesh,
                transform=outer @ node.matrix,
                object_id=object_id,
            )
            meshes.append(mesh)
            objects.append(obj)

    return ImportResult(
        objects=objects,
        meshes=meshes,
        materials=created_materials,
        warnings=warnings,
        source=document.source,
        texture_cache_dir=document.texture_cache_dir,
    )


__all__ = [
    "GltfDocument",
    "GltfMaterialData",
    "GltfNode",
    "GltfPrimitive",
    "load_gltf",
    "parse_gltf",
]
