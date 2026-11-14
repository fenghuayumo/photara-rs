"""Run in background Blender: shared mesh attributes must survive sync and edits.

This exercises AttributeSet::add_shared, which needs the embedded renderer's
own implicit-sharing callbacks even when host Cycles has already initialized.
"""
from pathlib import Path
import sys

import addon_utils
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
addon_utils.enable('photara_rs_engine')
from photara_rs_engine import native_backend

assert native_backend.available(), 'Requires the embedded Photara module'
scene = bpy.context.scene
scene.render.engine = 'PHOTARA_RS'
scene.render.resolution_x = scene.render.resolution_y = 32
scene.render.resolution_percentage = 100
scene.photara_rs.samples = 2
scene.photara_rs.denoise = False

# A triangle keeps point/corner attribute sizes compatible with zero-copy sync.
mesh = bpy.data.meshes.new('Shared attribute regression')
mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (0, 1, 0)], [], [(0, 1, 2)])
attribute = mesh.attributes.new('shared_uv', 'FLOAT2', 'POINT')
attribute.data.foreach_set('vector', [0, 0, 1, 0, 0.5, 1])
obj = bpy.data.objects.new('Shared attribute regression', mesh)
scene.collection.objects.link(obj)
material = bpy.data.materials.new('Shared attribute material')
material.use_nodes = True
nodes = material.node_tree.nodes
source = nodes.new('ShaderNodeAttribute')
source.attribute_name = 'shared_uv'
material.node_tree.links.new(source.outputs['Color'],
                             nodes['Principled BSDF'].inputs['Base Color'])
mesh.materials.append(material)

bpy.ops.render.render()
# Exercise release/re-import as well as the initial shared reference acquisition.
attribute.data.foreach_set('vector', [0.1, 0.1, 0.9, 0.1, 0.5, 0.9])
mesh.update()
bpy.ops.render.render()
print('BLENDER_SHARED_ATTRIBUTES_PASS', flush=True)
