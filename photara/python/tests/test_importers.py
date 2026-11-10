"""OBJ/MTL importer tests (parser first, then a small GPU smoke test)."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import photara_rs as pr
from photara_rs.importers import ObjImportError, load_obj, parse_obj

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


OBJ_TEXT = """\
mtllib materials.mtl
o Quad
v -1 -1 0
v 1 -1 0
v 1 1 0
v -1 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
usemtl Red
f 1/1/1 2/2/1 3/3/1 4/4/1

o Triangle
v -1 0 1
v 0 1 1
v 1 0 1
usemtl Blue
f -3 -2 -1
"""

MTL_TEXT = """\
newmtl Red
Kd 1 0 0
Ns 100
map_Kd texture.png

newmtl Blue
Kd 0 0 1
Ke 0.1 0 0
"""


class ObjParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.obj_path = os.path.join(self.root, "scene.obj")
        self.mtl_path = os.path.join(self.root, "materials.mtl")
        self.tex_path = os.path.join(self.root, "texture.png")
        if Image is not None:
            Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8)).save(self.tex_path)
        with open(self.obj_path, "w", encoding="utf-8") as handle:
            handle.write(OBJ_TEXT)
        with open(self.mtl_path, "w", encoding="utf-8") as handle:
            handle.write(MTL_TEXT)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parse_geometry_and_materials(self) -> None:
        model = parse_obj(self.obj_path)
        self.assertEqual(len(model.parts), 2)
        red = model.materials["Red"]
        blue = model.materials["Blue"]
        self.assertAlmostEqual(float(red.base_color[0]), 1.0, places=5)
        self.assertAlmostEqual(float(red.base_color[1]), 0.0, places=5)
        self.assertLess(red.roughness, 0.2)
        self.assertEqual(red.base_color_texture, self.tex_path)
        self.assertGreater(blue.emission_strength, 0.0)

        quad = next(part for part in model.parts if part.material_name == "Red")
        triangle = next(part for part in model.parts if part.material_name == "Blue")
        self.assertEqual(quad.indices.shape, (2, 3))
        self.assertEqual(quad.vertices.shape[0], 4)
        self.assertIsNotNone(quad.normals)
        self.assertIsNotNone(quad.uvs)
        self.assertEqual(triangle.indices.shape, (1, 3))
        self.assertIsNotNone(triangle.normals)
        self.assertIsNone(triangle.uvs)

    def test_uv_seams_duplicate_vertices(self) -> None:
        path = os.path.join(self.root, "seam.obj")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                "vt 0 0\nvt 1 0\nvt 0 1\nvt 0.5 0.5\n"
                "f 1/1 2/2 3/3\nf 1/4 2/2 3/3\n"
            )
        model = parse_obj(path)
        self.assertEqual(len(model.parts), 1)
        part = model.parts[0]
        self.assertEqual(part.vertices.shape[0], 4)
        self.assertIsNotNone(part.uvs)

    def test_invalid_indices_raise(self) -> None:
        path = os.path.join(self.root, "bad.obj")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 99\n")
        with self.assertRaises(ObjImportError):
            parse_obj(path)
        with self.assertRaises(ObjImportError):
            parse_obj(os.path.join(self.root, "missing.obj"))

    def test_scene_import_and_render(self) -> None:
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
            result = scene.import_obj(
                self.obj_path,
                location=(1.0, 2.0, 0.0),
                object_id_start=10,
            )
            self.assertEqual(len(result.objects), 2)
            self.assertEqual(len(result.meshes), 2)
            self.assertGreaterEqual(len(result.materials), 2)
            self.assertTrue(all(obj.object_id >= 10 for obj in result.objects))
            self.assertEqual(tuple(result.objects[0].location), (1.0, 2.0, 0.0))

            scene.camera.location = (4.0, 3.0, -5.0)
            scene.camera.target = (0.0, 1.0, 0.0)
            scene.lights.new_area(
                "key",
                position=(2.0, 4.0, -2.0),
                target=(0.0, 1.0, 0.0),
                strength=300.0,
            )
            frame = scene.render(seed=1)
            ids = set(np.unique(frame.object_id).astype(int).tolist())
            self.assertTrue({10, 11}.issubset(ids))
        finally:
            scene.close()


if __name__ == "__main__":
    unittest.main()
