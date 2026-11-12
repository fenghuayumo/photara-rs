"""Hardware OMM attachment, conservative alpha clipping and rebuild regression."""
import os
import tempfile
from pathlib import Path
import unittest
import numpy as np
import photara_rs as pr
try:
    from PIL import Image
except ImportError:
    Image = None


def make_scene(path):
    scene = pr.Scene(96, 64, samples=64, max_bounces=4, denoise=False,
                     adaptive_sampling=False)
    scene.camera.location = (0, 0, -4)
    scene.camera.target = (0, 0, 0)
    scene.world.color = (.2, .3, .5)
    material = scene.materials.new("leaf", base_color=(.3, .8, .1), roughness=1)
    material.base_color_texture = {"path": str(path), "colorspace": "data"}
    material.configure_alpha_clip(.5)
    vertices = [(-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0)]
    mesh = scene.meshes.new("leaf", vertices, [(0,1,2), (0,2,3)],
                            uvs=[(0,0),(1,0),(1,1),(0,1)], material=material)
    scene.objects.new("leaf", mesh)
    return scene, material, mesh


@unittest.skipIf(Image is None, "Pillow required for alpha fixtures")
class OMMTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "leaf.png"
        pixels = np.full((128, 128, 4), 255, dtype=np.uint8)
        pixels[:, :, 3] = 0
        pixels[40:72, 44:76, 3] = 255 # Asymmetric to verify alpha orientation.
        Image.fromarray(pixels).save(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_hardware_toggle_matches_reference_and_realtime(self):
        images = []
        old = os.environ.get("PHOTARA_DISABLE_OMM")
        try:
            for disabled in ("0", "1"):
                os.environ["PHOTARA_DISABLE_OMM"] = disabled
                scene, _, _ = make_scene(self.path)
                with scene:
                    reference = scene.render(seed=3).rgb.copy()
                    stats = scene.omm_stats()
                    if disabled == "1":
                        self.assertFalse(stats["enabled"])
                        self.assertEqual(stats["meshes"], 0)
                    elif stats["enabled"]:
                        self.assertEqual(stats["meshes"], 1)
                        self.assertGreater(stats["transparent_states"], 0)
                    scene.configure_render_mode("REALTIME", samples_per_frame=64)
                    realtime = scene.render(seed=3).rgb.copy()
                    self.assertTrue(np.isfinite(realtime).all())
                    images.append((reference, realtime))
            # Skipped transparent surfaces can change path RNG dimensions, so
            # compare energy within Monte Carlo noise rather than bit identity.
            for mode in (0,1):
                a,b = images[0][mode],images[1][mode]
                self.assertLess(np.sqrt(np.mean((a-b)**2)), .015)
        finally:
            if old is None: os.environ.pop("PHOTARA_DISABLE_OMM", None)
            else: os.environ["PHOTARA_DISABLE_OMM"] = old

    def test_alpha_uv_and_material_edits_invalidate_blas(self):
        scene, material, mesh = make_scene(self.path)
        with scene:
            scene.render()
            active = scene.omm_stats()["enabled"]
            if active: self.assertEqual(scene.omm_stats()["meshes"], 1)
            material.roughness = .7
            scene.render()
            if active: self.assertEqual(scene.omm_stats()["meshes"], 1)
            material.configure_alpha_clip(None)
            scene.render()
            self.assertEqual(scene.omm_stats()["meshes"], 0)
            material.configure_alpha_clip()
            scene.render()
            if active: self.assertEqual(scene.omm_stats()["meshes"], 1)
            # Collapsed UV triangle must not retain the original mask.
            mesh.update_positions(mesh.vertices, uvs=[(.5,.5)]*4)
            scene.render()
            self.assertEqual(scene.omm_stats()["meshes"], 0)
            mesh.update_positions(mesh.vertices, uvs=[(0,0),(1,0),(1,1),(0,1)])
            scene.render()
            if active: self.assertEqual(scene.omm_stats()["meshes"], 1)
            # Material replacement previously used a refit, which would retain stale OMMs.
            mesh.material = scene.materials.new("opaque", base_color=(.8,.1,.1))
            scene.render()
            self.assertEqual(scene.omm_stats()["meshes"], 0)

    def test_deep_cutout_stack_does_not_spend_transparency_bounces(self):
        previous = os.environ.get("PHOTARA_DISABLE_OMM")
        images = []
        try:
            for disabled in ("0", "1"):
                os.environ["PHOTARA_DISABLE_OMM"] = disabled
                scene, _, mesh = make_scene(self.path)
                with scene:
                    for i in range(16):
                        matrix = np.eye(4, dtype=np.float32)
                        matrix[2,3] = .04*(i+1)
                        scene.objects.new(str(i), mesh, transform=matrix)
                    wall = scene.materials.new("wall", base_color=(.7,.7,.7), roughness=1)
                    backdrop = scene.meshes.new("wall", [(-4,-4,2),(4,-4,2),(4,4,2),(-4,4,2)],
                                                 [(0,1,2),(0,2,3)], material=wall)
                    scene.objects.new("wall", backdrop)
                    scene.lights.new_area("front", position=(-1,1,-3), target=(0,0,2),
                                          size=(1,1), strength=40)
                    images.append(scene.render(seed=5).rgb.copy())
            # Traversal scheduling and stochastic texture filtering can alter
            # individual samples; compare convergence, not pixel identity.
            self.assertLess(float(np.sqrt(np.mean((images[0]-images[1])**2))), .001)
            self.assertLess(abs(float(images[0].mean()-images[1].mean())), .0003)
            self.assertGreater(float(images[0][8:20, 12:30].mean()), .1)
        finally:
            if previous is None: os.environ.pop("PHOTARA_DISABLE_OMM", None)
            else: os.environ["PHOTARA_DISABLE_OMM"] = previous

    def test_texture_replacement_and_async_guard(self):
        scene, material, _ = make_scene(self.path)
        with scene:
            scene.render()
            other = Path(self.tmp.name) / "opaque.png"
            Image.fromarray(np.full((32,32,4), 255, dtype=np.uint8)).save(other)
            material.base_color_texture = {"path": str(other), "colorspace": "data"}
            scene.render()
            self.assertEqual(scene.omm_stats()["meshes"], 0)
            material.base_color_texture = {"path": str(self.path), "colorspace": "data"}
            scene.render()
            if scene.omm_stats()["enabled"]: self.assertEqual(scene.omm_stats()["meshes"], 1)
            job = scene.render_async()
            with self.assertRaises(RuntimeError): material.configure_alpha_clip(None)
            job.wait()
            material.base_color_texture = None
            scene.render()
            self.assertEqual(scene.omm_stats()["meshes"], 0)


if __name__ == "__main__": unittest.main()
