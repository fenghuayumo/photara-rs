"""Real Blender/Photara integration checks against the installed add-on.

blender.exe --background --python-exit-code 1 --python scripts/test_blender.py
Use -- --viewport in a windowed Blender to also test asynchronous viewport edits.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import traceback

import bpy
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "test_outputs"
OUTPUT.mkdir(exist_ok=True)


def prepare_scene():
    import addon_utils

    addon_utils.enable("photara_rs_engine", default_set=True)
    assert "photara_rs_engine" in bpy.context.preferences.addons
    import photara_rs_engine
    error = photara_rs_engine.ensure_photara_module()
    if error:
        raise RuntimeError(error)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'PHOTARA_RS'
    scene.render.resolution_x = 192
    scene.render.resolution_y = 144
    scene.render.resolution_percentage = 100
    scene.photara_rs.samples = 8
    scene.photara_rs.viewport_samples = 4
    scene.photara_rs.denoise = False
    scene.photara_rs.adaptive_sampling = False
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.object
    cube.pass_index = 7
    material = bpy.data.materials.new("Photara test material")
    material.use_nodes = True
    material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.8, 0.03, 0.02, 1)
    cube.data.materials.append(material)
    bpy.ops.object.camera_add(location=(4, -6, 3))
    camera = bpy.context.object
    camera.rotation_euler = (-camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = camera
    bpy.ops.object.light_add(type='AREA', location=(1, -3, 5))
    light = bpy.context.object
    light.rotation_euler = (-light.location).to_track_quat('-Z', 'Y').to_euler()
    light.data.energy = 600
    light.data.size = 3
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.02, 0.02, 0.02, 1)
    scene.view_layers[0].use_pass_z = True
    bpy.context.view_layer.update()
    return scene, cube, material


def run_final(scene, cube, material):
    import photara_rs as pr
    from photara_rs_engine import sync
    from photara_rs_engine.engine import PhotaraRenderEngine

    captured = {}
    fill_result = PhotaraRenderEngine._fill_render_result
    def verify_passes(self, result, frame, view_layer):
        fill_result(self, result, frame, view_layer)
        for name, channels in (("Combined", 4), ("Depth", 1)):
            render_pass = result.layers[0].passes[name]
            values = np.empty(192 * 144 * channels, dtype=np.float32)
            render_pass.rect.foreach_get(values)
            captured[name] = values
    PhotaraRenderEngine._fill_render_result = verify_passes

    with pr.Scene(96, 72, samples=4, denoise=False, adaptive_sampling=False) as native:
        state = sync.SceneSyncState(native)
        state.sync(bpy.context.evaluated_depsgraph_get())
        state.sync_camera(scene.camera, 96 / 72)
        projection = scene.camera.calc_matrix_camera(
            bpy.context.evaluated_depsgraph_get(), x=96, y=72)
        assert abs(native.camera.fov - sync.fov_from_projection_degrees(projection, 96 / 72)) < 1e-4
        first = native.render()
        assert np.isfinite(first.beauty).all()
        assert np.count_nonzero(first.object_id == 7) > 100, "Missing visible cube"
        assert float(first.beauty[..., 0].max()) > 0.05, "Area light did not illuminate cube"
        # Shared light data must still produce two independent light instances.
        light = next(obj for obj in scene.objects if obj.type == 'LIGHT')
        duplicate = light.copy()
        scene.collection.objects.link(duplicate)
        duplicate.location.x += 2
        # Real collection instances must not collapse to their source object.
        collection = bpy.data.collections.new("Photara instance source")
        source = cube.copy()
        collection.objects.link(source)
        for x in (-3, 3):
            instance = bpy.data.objects.new("Photara collection instance", None)
            instance.instance_type = 'COLLECTION'
            instance.instance_collection = collection
            instance.location.x = x
            scene.collection.objects.link(instance)
        bpy.context.view_layer.update()
        state.sync(bpy.context.evaluated_depsgraph_get())
        assert len(state.lights) == 2, "Linked lights collapsed"
        assert len(state.objects) == 3, "Collection instances collapsed"
        empty = bpy.data.objects.new("Photara empty mesh", bpy.data.meshes.new("Empty"))
        scene.collection.objects.link(empty)
        bpy.context.view_layer.update()
        state.sync(bpy.context.evaluated_depsgraph_get())
        assert len(state.objects) == 3, "Empty mesh should be ignored"
        assert len(state.lights) == 2
        cube.location.x = 0.7
        material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.02, 0.8, 0.03, 1)
        bpy.context.view_layer.update()
        state.sync(bpy.context.evaluated_depsgraph_get())
        changed = native.render()
        assert not np.allclose(first.beauty, changed.beauty), "Scene edits were ignored"
        bpy.data.objects.remove(duplicate, do_unlink=True)
        bpy.context.view_layer.update()
        state.sync(bpy.context.evaluated_depsgraph_get())
        assert len(state.lights) == 1
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(OUTPUT / "photara_blender_verified.png")
    bpy.ops.render.render(write_still=True)
    assert set(captured) == {"Combined", "Depth"}, "Missing render passes"
    assert np.isfinite(captured['Combined']).all()
    assert np.count_nonzero((captured['Depth'] > 0) & (captured['Depth'] < 1e9)) > 100
    assert Path(scene.render.filepath).stat().st_size > 1000
    # Save the passes as a multilayer EXR. Read the PNG for display validation;
    # multilayer images do not expose a flat image.pixels buffer in all builds.
    if hasattr(scene.render.image_settings, 'media_type'):
        scene.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
    scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    exr = OUTPUT / "photara_blender_verified.exr"
    bpy.data.images['Render Result'].save_render(str(exr), scene=scene)
    assert exr.stat().st_size > 1000
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    pixels = np.array(image.pixels[:], dtype=np.float32)
    assert pixels.size == 192 * 144 * 4 and np.isfinite(pixels).all()
    assert pixels.reshape(-1, 4)[:, :3].std() > 0.01, "Empty Blender render result"
    bpy.data.images.remove(image)
    if hasattr(scene.render.image_settings, 'media_type'):
        scene.render.image_settings.media_type = 'IMAGE'
    scene.render.image_settings.file_format = 'PNG'
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT / "photara_demo.blend"))
    report = {"blender": bpy.app.version_string, "python": sys.version,
              "package": pr.__file__, "final_render": "passed", "sync_edits": "passed",
              "collection_instances": "passed", "linked_lights": "passed",
              "combined_depth": "passed", "empty_mesh": "passed", "camera_fov": "passed"}
    (OUTPUT / "blender_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PHOTARA_BLENDER_TEST_OK", flush=True)


def run_viewport(scene, cube, material):
    from photara_rs_engine import engine

    # Observe the actual engine callbacks, GPU upload and texture draw.
    counts = {"uploads": 0, "draws": 0, "sample_levels": []}
    upload = engine.ViewportState._upload_result
    draw = engine.ViewportState.draw
    invalidate = engine.ViewportState.invalidate
    def checked_invalidate(self, depsgraph=None, **kwargs):
        invalidate(self, depsgraph, **kwargs)
        assert self.pending_frame is None, "Pending frames from the previous view must be discarded"
    def checked_upload(self, frame):
        assert not self.gpu_display, "Native GPU display must never upload CPU pixels"
        upload(self, frame)
        assert self.texture is not None
        assert frame.beauty.shape[:2] == (self.height, self.width), "Viewport must stay at full resolution"
        counts["uploads"] += 1
        counts["sample_levels"].append(self.pending_samples)
        counts["settled"] = (self.pending_samples == 6 and frame.beauty.shape[:2] == (self.height, self.width))
        counts["material_updated"] = any(
            np.allclose(mat.base_color, (0.02, 0.8, 0.03)) for mat in self.scene.materials)
    def checked_draw(self, context):
        draw(self, context)
        if self.gpu_display:
            import _cycles
            stats = _cycles.photara_display_stats(self._display_capsule)
            counts['gpu_updates'] = stats['gpu_updates']
            counts['cpu_maps'] = stats['cpu_maps']
            assert stats['cpu_maps'] == 0, "Display silently fell back to CPU mapping"
            if stats['gpu_updates']:
                counts['draws'] += 1
            counts['settled'] = (not self.needs_update and self.job is None
                                  and self.current_samples == 6)
            counts['material_updated'] = any(
                np.allclose(mat.base_color, (0.02,0.8,0.03)) for mat in self.scene.materials)
        elif self.texture is not None:
            counts["draws"] += 1
    failures = []
    engine.ViewportState._upload_result = checked_upload
    engine.ViewportState.draw = checked_draw
    engine.ViewportState.invalidate = checked_invalidate
    engine.PhotaraRenderEngine._report_viewport_failure = lambda self, exc: failures.append(str(exc))
    area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
    area.spaces.active.region_3d.view_perspective = 'PERSP'
    area.spaces.active.shading.type = 'RENDERED'
    start = time.monotonic()
    stage = 0
    def poll():
        nonlocal stage
        try:
            assert not failures, failures
            if time.monotonic() - start > 90:
                raise AssertionError(f"Viewport timed out: {counts}")
            if (counts.get('uploads', 0) >= 1 or counts.get('gpu_updates', 0) >= 1) and stage == 0:
                stage = 1
                counts['before_edit'] = counts['uploads']
                cube.location.z += 0.3
                material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.02, 0.8, 0.03, 1)
                cube.data.vertices[0].co.x += 0.15
                cube.data.update()
                scene.photara_rs.max_bounces = 3
                scene.photara_rs.viewport_samples = 6
                area.spaces.active.region_3d.view_distance *= 1.1
                bpy.context.view_layer.update()
                area.tag_redraw()
            elif (stage == 1 and counts.get('settled') and counts.get('material_updated')
                  and counts['draws'] >= 3):
                print("PHOTARA_VIEWPORT_TEST_OK", json.dumps(counts), flush=True)
                (OUTPUT / "viewport_verification.json").write_text(json.dumps(counts), encoding='utf-8')
                bpy.ops.wm.quit_blender()
                return None
            return 0.1
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
            # Exit nonzero so CI cannot mistake Blender's normal exit for success.
            import os
            os._exit(1)
    bpy.app.timers.register(poll, first_interval=0.2)


if __name__ == '__main__':
    scene, cube, material = prepare_scene()
    if '--viewport' in sys.argv:
        run_viewport(scene, cube, material)
    else:
        run_final(scene, cube, material)
