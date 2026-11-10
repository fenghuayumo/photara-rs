"""glTF/GLB importer tests."""

from __future__ import annotations

import base64
import json
import os
import struct
import tempfile
import unittest

import numpy as np

import photara_rs as pr
from photara_rs.importers import GltfImportError, parse_gltf

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


_GLB_MAGIC = 0x46546C67
_GLB_JSON = 0x4E4F534A
_GLB_BIN = 0x004E4942


def _triangle_arrays():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)
    return vertices, indices


def _triangle_document(*, embedded_image: bytes | None = None, with_texture: bool = False):
    vertices, indices = _triangle_arrays()
    payload = vertices.tobytes() + indices.tobytes()
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": vertices.nbytes},
        {"buffer": 0, "byteOffset": vertices.nbytes, "byteLength": indices.nbytes},
    ]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
        {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
    ]
    document = {
        "asset": {"version": "2.0"},
        "buffers": [
            {
                "byteLength": len(payload),
                "uri": "data:application/octet-stream;base64,"
                + base64.b64encode(payload).decode(),
            }
        ],
        "bufferViews": views,
        "accessors": accessors,
        "materials": [
            {
                "name": "Red",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 0.0, 0.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.5,
                },
            }
        ],
        "meshes": [
            {
                "name": "Tri",
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "material": 0}
                ],
            }
        ],
        "nodes": [{"name": "TriNode", "mesh": 0, "translation": [1.0, 2.0, 0.0]}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    if with_texture:
        if embedded_image is not None:
            image_offset = len(payload)
            payload += embedded_image
            views.append(
                {"buffer": 0, "byteOffset": image_offset, "byteLength": len(embedded_image)}
            )
            document["buffers"][0]["byteLength"] = len(payload)
            document["buffers"][0]["uri"] = (
                "data:application/octet-stream;base64," + base64.b64encode(payload).decode()
            )
            document["images"] = [{"bufferView": 2, "mimeType": "image/png"}]
        else:
            document["images"] = [{"uri": "texture.png"}]
        document["textures"] = [{"source": 0, "sampler": 0}]
        document["samplers"] = [
            {"wrapS": 33071, "wrapT": 33071, "magFilter": 9728, "minFilter": 9728}
        ]
        document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {
            "index": 0,
            "texCoord": 0,
        }
    return document, payload


def _write_glb(path: str, document: dict, bin_data: bytes) -> None:
    json_data = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_data += b" " * ((-len(json_data)) % 4)
    bin_data += b"\x00" * ((-len(bin_data)) % 4)
    total = 12 + 8 + len(json_data) + 8 + len(bin_data)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        handle.write(struct.pack("<II", len(json_data), _GLB_JSON))
        handle.write(json_data)
        handle.write(struct.pack("<II", len(bin_data), _GLB_BIN))
        handle.write(bin_data)


class GltfParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parse_triangle_and_material(self) -> None:
        document, _ = _triangle_document()
        path = os.path.join(self.root, "tri.gltf")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        model = parse_gltf(path)
        self.assertEqual(len(model.nodes), 1)
        self.assertEqual(len(model.nodes[0].primitives), 1)
        np.testing.assert_allclose(model.nodes[0].matrix[:3, 3], [1.0, 2.0, 0.0])
        primitive = model.nodes[0].primitives[0]
        self.assertEqual(primitive.vertices.shape, (3, 3))
        self.assertEqual(primitive.indices.tolist(), [[0, 1, 2]])
        self.assertIsNotNone(primitive.normals)
        self.assertIsNone(primitive.uvs)
        self.assertEqual(model.materials[0].base_color, (1.0, 0.0, 0.0))

    def test_column_major_matrix(self) -> None:
        document, _ = _triangle_document()
        document["nodes"][0].pop("translation")
        document["nodes"][0]["matrix"] = [
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            4,
            5,
            6,
            1,
        ]
        path = os.path.join(self.root, "matrix.gltf")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        model = parse_gltf(path)
        np.testing.assert_allclose(model.nodes[0].matrix[:3, 3], [4.0, 5.0, 6.0])

    def test_hierarchy_and_multiple_primitives(self) -> None:
        document, _ = _triangle_document()
        document["materials"].append(
            {"name": "Blue", "pbrMetallicRoughness": {"baseColorFactor": [0.0, 0.0, 1.0, 1.0]}}
        )
        document["meshes"][0]["primitives"].append(
            {"attributes": {"POSITION": 0}, "indices": 1, "material": 1}
        )
        document["nodes"] = [
            {"name": "Parent", "translation": [1.0, 0.0, 0.0], "children": [1]},
            {"name": "Child", "mesh": 0, "translation": [0.0, 2.0, 0.0]},
        ]
        document["scenes"] = [{"nodes": [0]}]
        path = os.path.join(self.root, "hierarchy.gltf")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        model = parse_gltf(path)
        self.assertEqual(len(model.nodes), 1)
        node = model.nodes[0]
        self.assertEqual(node.name, "Child")
        np.testing.assert_allclose(node.matrix[:3, 3], [1.0, 2.0, 0.0])
        self.assertEqual(len(node.primitives), 2)
        self.assertEqual(
            [primitive.material_index for primitive in node.primitives], [0, 1]
        )

    def test_glb(self) -> None:
        document, payload = _triangle_document()
        del document["buffers"][0]["uri"]
        path = os.path.join(self.root, "tri.glb")
        _write_glb(path, document, payload)
        model = parse_gltf(path)
        self.assertEqual(model.nodes[0].primitives[0].vertices.shape, (3, 3))

    def test_external_texture_options(self) -> None:
        document, _ = _triangle_document(with_texture=True)
        with open(os.path.join(self.root, "texture.png"), "wb") as handle:
            handle.write(b"placeholder")
        path = os.path.join(self.root, "textured.gltf")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        model = parse_gltf(path)
        material = model.materials[0]
        self.assertEqual(material.base_color_texture, os.path.join(self.root, "texture.png"))
        self.assertEqual(material.texture_interpolation, "closest")
        self.assertEqual(material.texture_extension, "extend")

    def test_embedded_texture_extraction(self) -> None:
        if Image is None:
            self.skipTest("Pillow is not installed")
        image_path = os.path.join(self.root, "source.png")
        Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8)).save(image_path)
        with open(image_path, "rb") as handle:
            png = handle.read()
        document, _ = _triangle_document(embedded_image=png, with_texture=True)
        path = os.path.join(self.root, "embedded.gltf")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        model = parse_gltf(path)
        texture = model.materials[0].base_color_texture
        self.assertIsNotNone(texture)
        self.assertTrue(os.path.isfile(texture))
        self.assertTrue(texture.lower().endswith(".png"))
        self.assertIsNotNone(model.texture_cache_dir)

    def test_invalid_accessor_raises(self) -> None:
        document, _ = _triangle_document()
        document["accessors"][0]["count"] = 99
        path = os.path.join(self.root, "bad.gltf")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(GltfImportError):
            parse_gltf(path)

    def test_scene_import_and_render(self) -> None:
        document, _ = _triangle_document()
        path = os.path.join(self.root, "render.gltf")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        try:
            scene = pr.Scene(
                64,
                48,
                samples=2,
                max_bounces=2,
                denoise=False,
                adaptive_sampling=False,
                device="OPTIX",
            )
        except RuntimeError as exc:
            self.skipTest(f"OptiX is unavailable: {exc}")
        try:
            result = scene.import_gltf(
                path, location=(1.0, 2.0, 0.0), object_id_start=20
            )
            self.assertEqual(len(result.objects), 1)
            self.assertEqual(len(result.materials), 1)
            np.testing.assert_allclose(result.objects[0].location, [2.0, 4.0, 0.0])
            scene.camera.location = (2.5, 4.5, -3.0)
            scene.camera.target = (2.5, 4.5, 0.0)
            scene.lights.new_area(
                "key",
                position=(2.5, 5.0, -2.0),
                target=(2.5, 4.5, 0.0),
                strength=100.0,
            )
            frame = scene.render(seed=1)
            self.assertIn(20, set(np.unique(frame.object_id).astype(int).tolist()))
        finally:
            scene.close()


if __name__ == "__main__":
    unittest.main()
