"""Actual MIP selection and mode isolation on the native renderer.

OpenImageIO is used only to create a diagnostic texture whose mip levels have
known different colors, so these tests exercise LOD rather than base-level RGB.
"""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import photara_rs as pr
try:
    import OpenImageIO as oiio
except ImportError:
    oiio = None


def write_mip_texture(path):
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(oiio.geterror())
    try:
        for level in range(11):
            size = max(1, 1024 >> level)
            spec = oiio.ImageSpec(size, size, 3, oiio.FLOAT)
            spec.tile_width = spec.tile_height = 64
            spec.attribute("textureformat", "Plain Texture")
            spec.attribute("oiio:ColorSpace", "Linear")
            if not out.open(str(path), spec, "Create" if level == 0 else ("AppendMIPLevel" if out.supports("mipmap") else "AppendSubimage")):
                raise RuntimeError(out.geterror())
            pixels = np.full((size, size, 3), .04 + .08 * level, dtype=np.float32)
            if not out.write_image(pixels):
                raise RuntimeError(out.geterror())
    finally:
        out.close()


def plane_scene(path, singular_uv=False):
    scene = pr.Scene(96, 64, samples=16, max_bounces=2, denoise=False,
                     adaptive_sampling=False)
    scene.camera.location = (0, 1, -4)
    scene.camera.target = (0, 0, 4)
    scene.camera.fov = 45
    scene.world.color = (.5, .5, .5)
    matte = scene.materials.new("floor", base_color=(1, 1, 1), roughness=1)
    matte.base_color_texture = {"path": str(path), "colorspace": "data",
                               "interpolation": "linear", "extension": "repeat"}
    uv = [(0, 0), (0, 22), (8, 22), (8, 0)]
    if singular_uv:
        uv = [(0, 0), (0, 1), (0, 1), (0, 0)]
    mesh = scene.meshes.new("floor", [(-20, 0, -10), (-20, 0, 100),
                                      (20, 0, 100), (20, 0, -10)],
                            [(0, 1, 2), (0, 2, 3)], uvs=uv, material=matte)
    scene.objects.new("floor", mesh)
    return scene


class RayConeTests(unittest.TestCase):
    def test_validation_and_mode_persistence(self):
        with pr.Scene(32, 24, samples=4, denoise=False) as scene:
            self.assertEqual(scene.texture_lod, "RAY_CONES")
            for bad in ("invalid", "ray_cones", 0, None, []):
                with self.assertRaises(ValueError):
                    scene.configure_texture_lod(bad)
            scene.configure_texture_lod("RAY_DIFFERENTIALS")
            scene.configure_render_mode("REALTIME", samples_per_frame=1)
            self.assertEqual(scene.texture_lod, "RAY_DIFFERENTIALS")
            scene.configure_render_mode("REFERENCE")
            self.assertEqual(scene.texture_lod, "RAY_DIFFERENTIALS")
            job = scene.render_async()
            with self.assertRaises(RuntimeError):
                scene.configure_texture_lod()
            job.wait()

    @unittest.skipIf(oiio is None, "OpenImageIO is required for diagnostic mip texture")
    def test_grazing_mips_and_reference_realtime_switches(self):
        with tempfile.TemporaryDirectory() as tmp:
            texture = Path(tmp) / "levels.tx"
            write_mip_texture(texture)
            with plane_scene(texture) as scene:
                cone = scene.render(seed=3).albedo.copy()
                scene.configure_texture_lod("RAY_DIFFERENTIALS")
                legacy = scene.render(seed=3).albedo.copy()
                self.assertTrue(np.isfinite(cone).all())
                self.assertTrue(np.isfinite(legacy).all())
                # Grazing incidence must choose coarser (brighter-coded) mips.
                self.assertGreater(float((cone-legacy).mean()), .02)
                scene.configure_render_mode("REALTIME", samples_per_frame=4)
                scene.configure_texture_lod()
                realtime = scene.render(seed=3).albedo
                self.assertTrue(np.isfinite(realtime).all())
                self.assertGreater(float(realtime.mean()), 0)
                scene.resize(64, 48)
                self.assertTrue(np.isfinite(scene.render().rgb).all())
                scene.resize(96, 64)
                scene.configure_render_mode("REFERENCE")
                np.testing.assert_allclose(cone, scene.render(seed=3).albedo, atol=1e-5)
                scene.configure_texture_lod("RAY_DIFFERENTIALS")
                np.testing.assert_allclose(legacy, scene.render(seed=3).albedo, atol=1e-5)

    @unittest.skipIf(oiio is None, "OpenImageIO is required for PNG creation")
    def test_file_texture_builds_mip_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checker.png"
            pixels = np.zeros((256, 256, 3), dtype=np.uint8)
            pixels[::2, ::2] = 255
            pixels[1::2, 1::2] = 255
            out = oiio.ImageOutput.create(str(path))
            self.assertTrue(out.open(str(path), oiio.ImageSpec(256, 256, 3, oiio.UINT8)))
            self.assertTrue(out.write_image(pixels))
            out.close()
            with plane_scene(path) as scene:
                image = scene.render().albedo
                self.assertTrue(np.isfinite(image).all())
                self.assertTrue(list((Path(tmp) / "photara_tx").glob("*.tx")))

    @unittest.skipIf(oiio is None, "OpenImageIO is required for diagnostic mip texture")
    def test_singular_uv_falls_back_without_nans(self):
        with tempfile.TemporaryDirectory() as tmp:
            texture = Path(tmp) / "levels.tx"
            write_mip_texture(texture)
            with plane_scene(texture, singular_uv=True) as scene:
                cone = scene.render(seed=4).albedo.copy()
                scene.configure_texture_lod("RAY_DIFFERENTIALS")
                legacy = scene.render(seed=4).albedo
                self.assertTrue(np.isfinite(cone).all())
                np.testing.assert_allclose(cone, legacy, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
