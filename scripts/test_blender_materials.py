"""Run with Blender --background --python-exit-code 1 --python this_file."""
import sys
from pathlib import Path
from types import SimpleNamespace
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from test_blender import prepare_scene
from photara_rs_engine import sync
import photara_rs as pr

scene, cube, material = prepare_scene()
tree = material.node_tree
bsdf = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
# Put a disconnected shader first to catch node-order dependent export.
decoy = tree.nodes.new('ShaderNodeBsdfPrincipled')
decoy.inputs['Base Color'].default_value = (0, 0, 1, 1)
assert sync._surface_node(tree) == bsdf
outdir = ROOT / 'test_outputs'
outdir.mkdir(exist_ok=True)
image = bpy.data.images.new('Material regression', width=4, height=4)
image.pixels[:] = [0.03, 0.8, 0.02, 1] * 16
image.filepath_raw = str(outdir / 'material_green.png')
image.file_format = 'PNG'
image.save()
# Reload as FILE: generated and packed images need a separate native API.
image = bpy.data.images.load(image.filepath_raw, check_existing=False)
tex = tree.nodes.new('ShaderNodeTexImage')
tex.image = image
reroute = tree.nodes.new('NodeReroute')
tree.links.new(tex.outputs['Color'], reroute.inputs[0])
tree.links.new(reroute.outputs[0], bsdf.inputs['Base Color'])
with pr.Scene(96, 72, samples=8, denoise=False, adaptive_sampling=False) as native:
    state = sync.SceneSyncState(native)
    state.sync(bpy.context.evaluated_depsgraph_get())
    state.sync_camera(scene.camera, 96 / 72)
    exported = state.materials[material.as_pointer()]
    assert exported.base_color == (1, 1, 1)
    assert exported.base_color_texture['path'] == str(outdir / 'material_green.png')
    frame = native.render()
    mask = frame.object_id == 7
    mean = frame.beauty[mask, :3].mean(axis=0)
    assert mean[1] > mean[0] * 3 and mean[1] > mean[2] * 3, mean
    # Packed bytes must still render when the original external file is absent.
    image.pack()
    texture_path = Path(image.filepath)
    texture_path.unlink()
    sync.sync_material(state, material)
    packed_path = exported.base_color_texture['path']
    assert Path(packed_path).is_file()
    assert image.packed_file is not None
    assert sync._image_filepath(image) == packed_path
    frame = native.render()
    mean = frame.beauty[frame.object_id == 7, :3].mean(axis=0)
    assert mean[1] > mean[0] * 3, mean
    # One unsupported mapping must not prevent other scene materials syncing.
    coordinates = tree.nodes.new('ShaderNodeTexCoord')
    tree.links.new(coordinates.outputs['Generated'], tex.inputs['Vector'])
    sync.sync_material(state, material)
    assert exported.base_color_texture is None
    assert len(state._material_warnings) == 1
    sync.sync_material(state, material)
    assert len(state._material_warnings) == 1
    tree.links.remove(tex.inputs['Vector'].links[0])
    sync.sync_material(state, material)
    assert exported.base_color_texture is not None
    tree.links.remove(bsdf.inputs['Base Color'].links[0])
    bsdf.inputs['Base Color'].default_value = (0.8, 0.02, 0.01, 1)
    sync.sync_material(state, material)
    assert exported.base_color_texture is None
    frame = native.render()
    mean = frame.beauty[frame.object_id == 7, :3].mean(axis=0)
    assert mean[0] > mean[1] * 3, mean
    rgb = tree.nodes.new('ShaderNodeRGB')
    rgb.outputs[0].default_value = (0.1, 0.2, 0.9, 1)
    tree.links.new(rgb.outputs[0], bsdf.inputs['Emission Color'])
    sync.sync_material(state, material)
    assert np.allclose(exported.emission_color, (0.1, 0.2, 0.9))
print('MATERIAL_SYNC_PASS', flush=True)
