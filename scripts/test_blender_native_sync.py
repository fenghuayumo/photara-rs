"""Validate complete native shader/texture synchronization against Cycles.

Run in a disposable Blender process; never writes the input blend. Tests packed
generated images, legacy mixed closures, node groups, procedural textures,
bump, multiple material slots, and subsequent material/geometry edits.
"""
import json
from pathlib import Path
import sys

import addon_utils
import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'test_outputs/native_sync'
OUTPUT.mkdir(parents=True, exist_ok=True)
addon_utils.enable('photara_rs_engine')
from photara_rs_engine import engine, native_backend
import _photara_native
assert native_backend.available(), 'Embedded Photara renderer is unavailable'

scene = bpy.context.scene
scene.render.engine = 'PHOTARA_RS'
scene.render.resolution_x = 320
scene.render.resolution_y = 200
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.image_settings.color_depth = '16'
scene.render.film_transparent = False
scene.render.use_border = False
scene.render.use_persistent_data = True
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'
scene.view_settings.exposure = 0
scene.view_settings.gamma = 1
scene.photara_rs.samples = 32
scene.photara_rs.adaptive_sampling = False
scene.photara_rs.denoise = False
scene.photara_rs.max_bounces = 4
scene.photara_rs.seed = 0
scene.cycles.device = 'GPU'
scene.cycles.samples = 32
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 4
scene.cycles.diffuse_bounces = 4
scene.cycles.glossy_bounces = 4
scene.cycles.transmission_bounces = 4
scene.cycles.seed = 0
scene.cycles.use_pixel_jitter = True
scene.cycles.pixel_filter_type = 'BLACKMAN_HARRIS'
scene.cycles.filter_width = 1.5
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
for device in prefs.devices:
    device.use = device.type == 'OPTIX'

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
world = bpy.data.worlds.new('Native sync test world')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (.15, .15, .15, 1)
scene.world = world

image = bpy.data.images.new('Packed generated texture', width=16, height=16, alpha=True)
pixels = np.ones((16, 16, 4), np.float32)
yy, xx = np.indices((16,16))
pixels[..., :3] = np.where(((xx//4 + yy//4)%2)[...,None],
                          (0.9, .05, .02), (.02, .3, .9))
image.pixels.foreach_set(pixels.ravel())
image.pack()
mixed = bpy.data.materials.new('Legacy image diffuse glossy mix')
mixed.use_nodes = True
nodes = mixed.node_tree.nodes
nodes.clear()
output = nodes.new('ShaderNodeOutputMaterial')
diffuse = nodes.new('ShaderNodeBsdfDiffuse')
glossy = nodes.new('ShaderNodeBsdfAnisotropic')
glossy.inputs['Roughness'].default_value = .3
texture = nodes.new('ShaderNodeTexImage')
texture.image = image
mix = nodes.new('ShaderNodeMixShader')
mix.inputs[0].default_value = .2
links = mixed.node_tree.links
links.new(texture.outputs['Color'], diffuse.inputs['Color'])
links.new(diffuse.outputs[0], mix.inputs[1])
links.new(glossy.outputs[0], mix.inputs[2])
links.new(mix.outputs[0], output.inputs['Surface'])

group = bpy.data.node_groups.new('Nested procedural closure', 'ShaderNodeTree')
group.interface.new_socket(name='Surface', in_out='OUTPUT', socket_type='NodeSocketShader')
noise = group.nodes.new('ShaderNodeTexNoise')
noise.inputs['Scale'].default_value = 4
ramp = group.nodes.new('ShaderNodeValToRGB')
ramp.color_ramp.elements[0].color = (.02, .1, .6, 1)
ramp.color_ramp.elements[1].color = (.95, .6, .02, 1)
bsdf = group.nodes.new('ShaderNodeBsdfPrincipled')
bump = group.nodes.new('ShaderNodeBump')
bump.inputs['Strength'].default_value = .2
group_output = group.nodes.new('NodeGroupOutput')
group.links.new(noise.outputs['Fac'], ramp.inputs[0])
group.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
group.links.new(noise.outputs['Fac'], bump.inputs['Height'])
group.links.new(bump.outputs[0], bsdf.inputs['Normal'])
group.links.new(bsdf.outputs[0], group_output.inputs['Surface'])
procedural = bpy.data.materials.new('Procedural node group and bump')
procedural.use_nodes = True
group_node = procedural.node_tree.nodes.new('ShaderNodeGroup')
group_node.node_tree = group
procedural.node_tree.links.new(group_node.outputs[0],
                              procedural.node_tree.nodes['Material Output'].inputs['Surface'])

objects = []
for index, material in enumerate((mixed, procedural, mixed)):
    bpy.ops.mesh.primitive_cube_add(size=1.5, location=((index-1)*2, 0, 0))
    obj = bpy.context.object
    obj.rotation_euler = (.12, .18, .2)
    obj.data.materials.append(material)
    if index == 2:
        obj.data.materials.append(procedural)
        for face in obj.data.polygons:
            face.material_index = face.index % 2
    objects.append(obj)
bpy.ops.object.camera_add(location=(0, -9, 5))
camera = bpy.context.object
camera.rotation_euler = (-camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.lens = 42
scene.camera = camera
bpy.ops.object.light_add(type='AREA', location=(-2, -3, 6))
light = bpy.context.object
light.rotation_euler = (-light.location).to_track_quat('-Z','Y').to_euler()
light.data.energy = 800
light.data.shape = 'DISK'
light.data.size = 4

native_stats = []
original_render = engine.PhotaraRenderEngine.render
def observed_render(self, depsgraph):
    original_render(self, depsgraph)
    assert self._native, 'Legacy Python material conversion was used'
    native_stats.append(_photara_native.session_stats(self.session))
engine.PhotaraRenderEngine.render = observed_render


def render(name, label):
    scene.render.engine = name
    bpy.context.view_layer.update()
    path = OUTPUT / (label + '.png')
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    loaded = bpy.data.images.load(str(path), check_existing=False)
    data = np.asarray(loaded.pixels[:], np.float32).reshape(200,320,4).copy()
    bpy.data.images.remove(loaded)
    assert np.std(data[...,:3]) > .05, 'Blank render'
    return data


reports = []
for phase in ('initial', 'edited'):
    if phase == 'edited':
        ramp.color_ramp.elements[1].color = (.02, .9, .15, 1)
        objects[0].location.z += .2
        objects[2].data.vertices[0].co.x -= .2
        objects[2].data.update()
    native = render('PHOTARA_RS', 'photara_' + phase)
    reference = render('CYCLES', 'cycles_' + phase)
    error = float(np.sqrt(np.mean((native-reference)**2)))
    assert error < 1e-4, (phase, error)
    reports.append(dict(phase=phase, linear_rgba_rmse=error))
assert all(s and not s['use_resolution_divider'] and s['trace_width']==320
           and s['trace_height']==200 and s['image_texture_nodes']>0 for s in native_stats), native_stats
(OUTPUT/'verification.json').write_text(json.dumps(dict(comparisons=reports, native=native_stats),indent=2))
print('PHOTARA_NATIVE_SYNC_PASS',json.dumps(reports),flush=True)
