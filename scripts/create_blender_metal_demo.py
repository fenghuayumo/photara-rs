"""Create and render a Photara Metal studio scene with the installed Blender add-on.

Blender --background --python-exit-code 1 --python scripts/create_blender_metal_demo.py
Open test_outputs/blender_metal_demo/photara_metal_studio.blend and press F12.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import addon_utils
import bpy
from mathutils import Vector
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'test_outputs/blender_metal_demo'


def linear_color(hex_color):
    values = [int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
                 for v in values)


def material(name, color, *, roughness=0.4, metallic=0.0, transmission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get('Principled BSDF')
    node.inputs['Base Color'].default_value = (*linear_color(color), 1)
    node.inputs['Roughness'].default_value = roughness
    node.inputs['Metallic'].default_value = metallic
    node.inputs['Transmission Weight'].default_value = transmission
    node.inputs['IOR'].default_value = 1.46
    mat.diffuse_color = (*linear_color(color), 1)
    return mat


def move_to_collection(obj, collection):
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def finish(obj, name, mat, collection, *, bevel=0):
    obj.name = name
    move_to_collection(obj, collection)
    obj.data.materials.append(mat)
    if obj.type == 'MESH':
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        if bevel:
            mod = obj.modifiers.new('Soft manufactured edges', 'BEVEL')
            mod.width, mod.segments = bevel, 4
            obj.modifiers.new('Weighted normals', 'WEIGHTED_NORMAL')
    return obj


def plinth(name, position, radius, height, mat, collection):
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=radius, depth=height,
                                       location=(*position, height / 2))
    return finish(bpy.context.object, name, mat, collection, bevel=0.045)


def seat(obj, height):
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        lowest = min((evaluated.matrix_world @ vertex.co).z for vertex in mesh.vertices)
    finally:
        evaluated.to_mesh_clear()
    obj.location.z += height - lowest + 0.012


def area(name, position, target, energy, size, color, collection, *, size_y=None):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy, data.size = energy, size
    data.color = linear_color(color)
    if size_y:
        data.shape, data.size_y = 'RECTANGLE', size_y
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()
    return obj


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    addon_utils.enable('photara_rs_engine', default_set=True, persistent=True)
    import photara_rs_engine
    error = photara_rs_engine.ensure_photara_module()
    if error:
        raise RuntimeError(error)
    import photara_rs as pr
    from photara_rs_engine.engine import PhotaraRenderEngine
    from photara_rs_engine import native_backend
    if native_backend.available():
        raise RuntimeError('This demo validates the installed standalone Python/Metal add-on path')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.name = 'Photara | Metal Studio'
    groups = {}
    for name in ('01 · Stage', '02 · Plinths', '03 · Material studies', '04 · Softboxes'):
        groups[name] = bpy.data.collections.new(name)
        scene.collection.children.link(groups[name])
    stage, bases, forms, lights = groups.values()

    backdrop = material('Stage / midnight blue', '243A46', roughness=0.58)
    pedestal = material('Plinth / satin graphite', '172C37', roughness=0.3, metallic=0.12)
    rim = material('Trim / brushed champagne', 'BFA273', roughness=0.3, metallic=0.8)
    ceramic = material('Suzanne / glazed terracotta', 'D96E49', roughness=0.26)
    gold = material('Torus / champagne gold', 'E7BD75', roughness=0.2, metallic=1)
    glass = material('Sphere / clear optical glass', 'E6F4F8', roughness=0.035, transmission=1)

    # A seamless floor-to-wall sweep; plain Principled materials keep this demo
    # compatible with the add-on's standalone material converter.
    profile = [(-14, 0), (3, 0)]
    for i in range(1, 25):
        angle = math.pi * 0.5 * i / 24
        profile.append((3 + 2 * math.sin(angle), 2 - 2 * math.cos(angle)))
    profile.append((5, 9))
    vertices = [(x, y, z) for y, z in profile for x in (-14, 14)]
    faces = [(2*i, 2*i+1, 2*i+3, 2*i+2) for i in range(len(profile)-1)]
    mesh = bpy.data.meshes.new('Seamless cyclorama')
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new('Seamless blue cyclorama', mesh)
    stage.objects.link(obj)
    obj.data.materials.append(backdrop)
    for polygon in mesh.polygons:
        polygon.use_smooth = True

    for name, xy, radius, height in (
        ('Glass pedestal', (-2.25, -0.1), 0.98, 0.52),
        ('Hero pedestal', (0, 0.55), 1.22, 0.34),
        ('Gold pedestal', (2.25, 0.15), 0.98, 0.52),
    ):
        plinth(name + ' / trim', xy, radius * 1.006, 0.065, rim, bases)
        plinth(name, xy, radius, height, pedestal, bases)

    bpy.ops.mesh.primitive_monkey_add(location=(0, 0.55, 1.2))
    hero = finish(bpy.context.object, '01 / Ceramic Suzanne', ceramic, forms)
    hero.rotation_euler.z = math.radians(12)
    hero.scale = (0.93,) * 3
    subd = hero.modifiers.new('Sculpture smooth surface', 'SUBSURF')
    subd.levels = subd.render_levels = 2
    seat(hero, 0.34)
    hero.pass_index = 1

    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.74,
                                       location=(-2.25, -0.1, 1.28))
    sphere = finish(bpy.context.object, '02 / Optical glass sphere', glass, forms)
    sphere.pass_index = 2

    bpy.ops.mesh.primitive_torus_add(major_segments=96, minor_segments=32,
                                   major_radius=0.65, minor_radius=0.19,
                                   location=(2.25, 0.15, 1.5),
                                   rotation=(math.radians(73), math.radians(12), math.radians(-24)))
    torus = finish(bpy.context.object, '03 / Champagne gold torus', gold, forms)
    seat(torus, 0.52)
    torus.pass_index = 3

    area('Key / large warm softbox', (-3.5, -4.5, 6.5), (0, 0, 1),
         1900, 5, 'FFF1DF', lights, size_y=3)
    area('Rim / cool strip', (2.5, 3.5, 5), (0, 0.3, 1),
         2300, 3, 'D6E8FF', lights, size_y=4)
    area('Fill / right softbox', (5, -2.0, 3.8), (0, 0, 1),
         1050, 3, 'FFF4E6', lights, size_y=4)

    camera_data = bpy.data.cameras.new('Studio / 52 mm')
    camera = bpy.data.objects.new('Camera / final composition', camera_data)
    stage.objects.link(camera)
    camera.location = (6.8, -11.5, 5.4)
    target = Vector((0, 0.4, 1.05))
    camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera_data.lens = 52
    scene.camera = camera
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.025, 0.035, 0.05, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.25

    scene.render.engine = 'PHOTARA_RS'
    p = scene.photara_rs
    p.device, p.render_mode = 'METAL', 'REFERENCE'
    p.samples, p.viewport_samples, p.max_bounces = 128, 16, 12
    p.denoise, p.denoiser = True, 'OPENIMAGEDENOISE'
    p.denoise_min_samples = 0
    p.viewport_denoise = True
    p.adaptive_sampling = False
    scene.render.resolution_x, scene.render.resolution_y = 1000, 750
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(OUTPUT / 'photara_metal_studio.png')
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    scene.view_layers[0].use_pass_z = True
    scene['Photara demo'] = 'Metal · Reference · 128 SPP · OIDN · press F12 to render'

    for screen in bpy.data.screens:
        for area_ui in screen.areas:
            if area_ui.type == 'VIEW_3D':
                area_ui.spaces.active.region_3d.view_perspective = 'CAMERA'
                area_ui.spaces.active.overlay.show_overlays = False
            elif area_ui.type == 'PROPERTIES':
                area_ui.spaces.active.context = 'RENDER'
    bpy.ops.object.select_all(action='DESELECT')
    hero.select_set(True)
    bpy.context.view_layer.objects.active = hero
    bpy.context.view_layer.update()
    blend = OUTPUT / 'photara_metal_studio.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))

    capture = {}
    original = PhotaraRenderEngine._fill_render_result
    def verify(self, result, frame, layer):
        original(self, result, frame, layer)
        assert np.isfinite(frame.beauty).all()
        assert float(frame.rgb.std()) > 0.01
        capture['combined_shape'] = list(frame.beauty.shape)
        capture['depth_valid_pixels'] = int(np.count_nonzero(frame.depth_valid))
    PhotaraRenderEngine._fill_render_result = verify
    started = time.perf_counter()
    try:
        bpy.ops.render.render(write_still=True)
    finally:
        PhotaraRenderEngine._fill_render_result = original
    if not capture or not Path(scene.render.filepath).is_file():
        raise RuntimeError('Photara add-on did not produce a valid Blender render')
    report = dict(blender=bpy.app.version_string, python=sys.version,
                  engine=scene.render.engine, device=p.device, samples=p.samples,
                  render_seconds=time.perf_counter()-started,
                  addon=photara_rs_engine.__file__, package=pr.__file__,
                  blend=str(blend), image=scene.render.filepath, **capture)
    (OUTPUT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PHOTARA_METAL_BLENDER_DEMO_OK', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
