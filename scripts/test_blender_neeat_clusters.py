"""Native mesh-light NEE-AT energy, instance and 2M-entry GPU regressions.

Run in factory-startup background Blender; --stress builds 2,097,152 emissive
triangles. The normal run compares CPU/GPU feedback against a Reference image.
"""
import json
import math
import os
from pathlib import Path
import sys

import addon_utils
import bpy
import numpy as np

addon_utils.enable('photara_rs_engine')
from photara_rs_engine import native_engine
import _photara_native

STRESS = '--stress' in sys.argv
OUT = Path(__file__).resolve().parents[1] / 'test_outputs' / 'neeat_clusters'
OUT.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = 'PHOTARA_RS'
scene.render.resolution_x = scene.render.resolution_y = 16 if STRESS else 48
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_depth = '32'
scene.render.filepath = str(OUT / 'current.exr')
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
p = scene.photara_rs
p.denoise = False
p.adaptive_sampling = False
p.history_samples = 0
p.realtime_warmup_frames = 2 if STRESS else 16
p.max_bounces = 2

material = bpy.data.materials.new('emitter')
material.use_nodes = True
nodes = material.node_tree.nodes
nodes['Principled BSDF'].inputs['Base Color'].default_value = (0, 0, 0, 1)
nodes['Principled BSDF'].inputs['Emission Color'].default_value = (.7, .5, .3, 1)
nodes['Principled BSDF'].inputs['Emission Strength'].default_value = 10


def emissive_grid(side):
    axis = np.linspace(-.3, .3, side + 1, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    vertices = np.stack((x, y, np.zeros_like(x)), axis=-1).reshape(-1, 3)
    base = (np.arange(side)[:, None] * (side + 1) + np.arange(side)[None, :]).ravel()
    faces = np.stack((base, base + side + 1, base + 1,
                      base + 1, base + side + 1, base + side + 2), axis=1).reshape(-1, 3)
    mesh = bpy.data.meshes.new('shared emissive grid')
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set('co', vertices.ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set('vertex_index', faces.astype(np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set('loop_start', np.arange(len(faces), dtype=np.int32) * 3)
    mesh.polygons.foreach_set('loop_total', np.full(len(faces), 3, dtype=np.int32))
    mesh.materials.append(material)
    mesh.update()
    return mesh


mesh = emissive_grid(1024 if STRESS else 24)
instances = 1 if STRESS else 8
for i in range(instances):
    obj = bpy.data.objects.new('emitter instance', mesh)
    scene.collection.objects.link(obj)
    angle = 2 * math.pi * i / instances
    obj.location = (2.5, 0, 2) if STRESS else (2.5 * math.cos(angle), 2.5 * math.sin(angle), 2)

bpy.ops.mesh.primitive_plane_add(size=20)
floor = bpy.context.object
diffuse = bpy.data.materials.new('diffuse receiver')
diffuse.use_nodes = True
diffuse.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (.7, .7, .7, 1)
diffuse.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 1
floor.data.materials.append(diffuse)

bpy.ops.object.camera_add(location=(0, 0, 4))
scene.camera = bpy.context.object
# A Blender camera points along local -Z by default.
scene.camera.rotation_euler = (0, 0, 0)
scene.camera.data.lens = 50
bpy.context.view_layer.update()
reports = []
original_render = native_engine.render


def capture(engine, depsgraph):
    original_render(engine, depsgraph)
    reports.append(dict(_photara_native.session_stats(engine.session)))


native_engine.render = capture


def render():
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    pixels = np.empty(scene.render.resolution_x * scene.render.resolution_y * 4, np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    rgb = pixels.reshape(-1, 4)[:, :3]
    assert np.isfinite(rgb).all()
    return rgb


if STRESS:
    p.render_mode, p.samples, p.realtime_neeat = 'REALTIME', 1, True
    p.realtime_neeat_mode = 'PURE'
    image = render()
    assert image.mean() > .001, '2M-light proposal failed to illuminate the receiver'
    assert reports[-1]['neeat_emitters'] == 2097152, reports[-1]
    assert reports[-1]['neeat_clusters'] == 4096, reports[-1]
    assert reports[-1]['neeat_tree_fraction'] == 0.0, reports[-1]
    assert reports[-1]['neeat_device_bytes'] < 72 * 1024 * 1024, reports[-1]
    print('NEEAT_2M_GPU_PASS', json.dumps(reports[-1]), flush=True)
    (OUT / 'stress.json').write_text(json.dumps(reports[-1], indent=2))
else:
    p.seed = 0
    p.render_mode, p.samples = 'REFERENCE', 4096
    reference = render()
    assert float(reference.mean()) > .001
    assert reports[-1]['neeat_emitters'] == 0
    results = {}
    for backend in ('GPU_HYBRID', 'GPU_PURE', 'CPU_PURE'):
        if backend.startswith('CPU'):
            os.environ['PHOTARA_NEEAT_CPU'] = '1'
        else:
            os.environ.pop('PHOTARA_NEEAT_CPU', None)
        p.render_mode, p.samples, p.realtime_neeat = 'REALTIME', 16, True
        p.realtime_neeat_mode = 'PURE' if backend.endswith('PURE') else 'HYBRID'
        images = []
        for seed in range(8):
            p.seed = seed + 100
            images.append(render())
            assert reports[-1]['neeat_emitters'] == instances * len(mesh.polygons), reports[-1]
            assert reports[-1]['neeat_clusters'] <= 4096, reports[-1]
            assert reports[-1]['neeat_tree_fraction'] == (0.0 if backend.endswith('PURE') else .25)
        mean = np.mean(images, axis=0)
        relative_bias = float((mean.mean() - reference.mean()) / reference.mean())
        assert abs(relative_bias) < .08, (backend, relative_bias)
        results[backend] = dict(relative_bias=relative_bias, stats=reports[-1])
        print('NEEAT_MESH_ENERGY_PASS', backend, json.dumps(results[backend]), flush=True)
    p.render_mode, p.samples = 'REFERENCE', 4096
    p.seed = 0
    repeat = render()
    np.testing.assert_allclose(reference, repeat, rtol=1e-5, atol=1e-6)
    (OUT / 'energy.json').write_text(json.dumps(results, indent=2))
    print('NEEAT_INSTANCE_REFERENCE_ISOLATION_PASS', flush=True)
