"""Blender background exporter for the supplied Sponza fixture and Photara XML.

Flattens shader groups, exports evaluated triangles, corner normals/UVs, images,
world and sun. Fails on unknown connected shader nodes instead of dropping them.
"""
import bpy
import json
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test_outputs/sponza_viewer'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'textures').mkdir(exist_ok=True)
source = (ROOT / 'src/scene/shader_nodes.cpp').read_text()
schemas = {}
for match in re.finditer(r'NODE_DEFINE\((\w+)\)\s*\{(.*?)\n\}', source, re.S):
    body = match[2]
    tag = re.search(r'NodeType::add\("([^"]+)"', body)
    if tag:
        schemas[tag[1]] = re.findall(r'SOCKET_(IN_|OUT_)?(\w+)\(\s*(\w+),\s*"([^"]+)"', body)

mapping = dict(ShaderNodeBsdfDiffuse='diffuse_bsdf', ShaderNodeBsdfAnisotropic='glossy_bsdf',
               ShaderNodeMixShader='mix_closure', ShaderNodeAddShader='add_closure',
               ShaderNodeBsdfTransparent='transparent_bsdf', ShaderNodeBsdfTranslucent='translucent_bsdf',
               ShaderNodeBsdfPrincipled='principled_bsdf', ShaderNodeTexImage='image_texture',
               ShaderNodeLightPath='light_path', ShaderNodeLayerWeight='layer_weight',
               ShaderNodeNormalMap='normal_map', ShaderNodeMath='math', ShaderNodeRGBCurve='rgb_curves',
               ShaderNodeMix='mix_color', ShaderNodeValToRGB='rgb_ramp', ShaderNodeBump='bump',
               ShaderNodeGamma='gamma', ShaderNodeNewGeometry='geometry', ShaderNodeInvert='invert',
               ShaderNodeFresnel='fresnel', ShaderNodeValue='value', ShaderNodeTexCoord='texture_coordinate',
               ShaderNodeBackground='background_shader', ShaderNodeTexSky='sky_texture')

def fmt(v):
    if isinstance(v, bool): return str(v).lower()
    if isinstance(v, str): return v
    try: return ' '.join(format(float(x), '.9g') for x in v)
    except TypeError: return format(float(v), '.9g')

def matrix(m):
    return fmt([m[r][c] for c in range(4) for r in range(4)])

images = {}
def image_path(image):
    if image.name not in images:
        path = OUT / 'textures' / (str(len(images)) + '.png')
        original = Path(bpy.path.abspath(image.filepath))
        if image.packed_file or not original.is_file():
            image.save_render(str(path), scene=bpy.context.scene)
        else:
            import shutil
            path = path.with_suffix(original.suffix)
            shutil.copy2(original, path)
        images[image.name] = path.relative_to(OUT).as_posix()
    return images[image.name]

def shader(parent, tree, output):
    cache = {}
    def resolve(sock, stack):
        if not sock.is_linked: return ('value', getattr(sock, 'default_value', 0))
        link = sock.links[0]
        node, socket = link.from_node, link.from_socket
        if node.type == 'REROUTE': return resolve(node.inputs[0], stack)
        if node.type == 'GROUP':
            group_out = next(n for n in node.node_tree.nodes if n.type == 'GROUP_OUTPUT' and n.is_active_output)
            target = next(s for s in group_out.inputs if s.identifier == socket.identifier)
            return resolve(target, stack + (node,))
        if node.type == 'GROUP_INPUT':
            target = next(s for s in stack[-1].inputs if s.identifier == socket.identifier)
            return resolve(target, stack[:-1])
        return emit(node, socket, stack)

    def socket_name(node, socket, direction, tag):
        candidates = [(typ, name, label) for d, typ, name, label in schemas[tag] if d == direction]
        index = list(node.outputs if direction == 'OUT_' else node.inputs).index(socket)
        if node.type in {'MIX_SHADER', 'ADD_SHADER'}:
            return ('closure', 'Closure') if direction == 'OUT_' else (('fac', 'Fac') if node.type == 'MIX_SHADER' and index == 0 else ('closure' + str(index if node.type == 'MIX_SHADER' else index + 1), 'Closure' + str(index if node.type == 'MIX_SHADER' else index + 1)))
        if node.type == 'MATH' and direction == 'IN_':
            return next((name, label) for typ, name, label in candidates if label == ['Value1', 'Value2', 'Value3'][index])
        aliases = {'Anisotropic': 'Anisotropy', 'Anisotropic Rotation': 'Rotation', 'Weight': 'SurfaceMixWeight'}
        label = aliases.get(socket.name, socket.name)
        if label == 'Factor' and any(t == 'Fac' for _, _, t in candidates): label = 'Fac'
        for typ, name, text in candidates:
            if text.lower().replace(' ', '') == label.lower().replace(' ', ''): return name, text
        if direction == 'OUT_' and len(candidates) == 1: return candidates[0][1:]
        return None

    def emit(node, socket, stack):
        tag = mapping.get(node.bl_idname)
        if tag is None: raise ValueError('Unsupported connected node: ' + node.bl_idname)
        key = (tuple(n.as_pointer() for n in stack), node.as_pointer())
        if key not in cache:
            name = 'n' + str(len(cache))
            cache[key] = name
            element = ET.SubElement(parent, 'background' if tag == 'background_shader' else tag, name=name)
            attrs = {n: t for d, t, n, label in schemas[tag] if not d}
            for attr in attrs:
                if hasattr(node, attr) and attr not in {'name'}:
                    value = getattr(node, attr)
                    if isinstance(value, (str, int, float, bool)) or hasattr(value, '__len__'):
                        element.set(attr, fmt(value.lower() if isinstance(value, str) else value))
            if tag == 'image_texture':
                element.set('filename', image_path(node.image))
                element.set('colorspace', node.image.colorspace_settings.name)
                element.set('extension', {'REPEAT': 'periodic', 'EXTEND': 'clamp', 'CLIP': 'black', 'MIRROR': 'mirror'}[node.extension])
            if tag == 'value': element.set('value', fmt(node.outputs[0].default_value))
            if tag == 'sky_texture':
                element.set('sky_type', {'NISHITA': 'multiple_scattering'}.get(node.sky_type, node.sky_type.lower()))
            if tag == 'mix_color':
                element.set('blend_type', node.blend_type.lower())
                element.set('use_clamp', fmt(node.clamp_factor))
            if tag == 'rgb_ramp':
                values = [node.color_ramp.evaluate(i / 255) for i in range(256)]
                element.set('ramp', fmt([x for v in values for x in v[:3]]))
                element.set('ramp_alpha', fmt([v[3] for v in values]))
            if tag == 'rgb_curves':
                node.mapping.initialize()
                vals = []
                for i in range(256):
                    for c in range(3):
                        v = node.mapping.evaluate(node.mapping.curves[3], i / 255)
                        vals.append(node.mapping.evaluate(node.mapping.curves[c], v))
                element.set('curves', fmt(vals))
            for sock in node.inputs:
                if not sock.enabled: continue
                target = socket_name(node, sock, 'IN_', tag)
                if target is None:
                    if sock.is_linked: raise ValueError(f'Unknown input {node.bl_idname} {sock.name}')
                    continue
                value = resolve(sock, stack)
                if value[0] == 'link':
                    ET.SubElement(parent, 'connect', attrib={'from': value[1], 'to': name + ' ' + target[0]})
                elif sock.type != 'SHADER' and sock.name not in {'Normal', 'Tangent', 'Vector'}:
                    v = value[1]
                    if sock.type == 'RGBA': v = v[:3]
                    element.set(target[0], fmt(v))
        target = socket_name(node, socket, 'OUT_', tag)
        if target is None: raise ValueError('Unknown output ' + socket.name)
        return 'link', cache[key] + ' ' + target[0]

    result = resolve(output.inputs['Surface'], ())
    ET.SubElement(parent, 'connect', attrib={'from': result[1], 'to': 'output surface'})

root = ET.Element('cycles')
ET.SubElement(root, 'integrator', max_bounce='8', use_adaptive_sampling='false')
camera = bpy.context.scene.camera
tfm = ET.SubElement(root, 'transform', matrix=matrix(camera.matrix_world @ Matrix.Diagonal((1, 1, -1, 1))))
ET.SubElement(tfm, 'camera', width='1280', height='1280', type='perspective', fov=fmt(camera.data.angle))
for i, material in enumerate(bpy.data.materials):
    material['xml_name'] = 'mat' + str(i)
    if not material.use_nodes: raise ValueError('Non-node material ' + material.name)
    output = next(n for n in material.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output)
    shader(ET.SubElement(root, 'shader', name=material['xml_name']), material.node_tree, output)
world = bpy.context.scene.world
shader(ET.SubElement(root, 'background'), world.node_tree,
       next(n for n in world.node_tree.nodes if n.type == 'OUTPUT_WORLD'))
ET.SubElement(root, 'light', light_type='background', strength='1 1 1', use_mis='true')
dg = bpy.context.evaluated_depsgraph_get()
triangles = 0
objects = 0
for instance in dg.object_instances:
    obj = instance.object
    if obj.type == 'LIGHT':
        if obj.data.type != 'SUN': raise ValueError('Only sun is implemented for this fixture')
        light_shader = ET.SubElement(root, 'shader', name='sun_emission')
        ET.SubElement(light_shader, 'emission', name='emit', color='1 1 1', strength='1')
        ET.SubElement(light_shader, 'connect', attrib={'from': 'emit emission', 'to': 'output surface'})
        t = ET.SubElement(ET.SubElement(root, 'state', shader='sun_emission'), 'transform', matrix=matrix(instance.matrix_world))
        ET.SubElement(t, 'light', light_type='sun', strength=fmt([c * obj.data.energy for c in obj.data.color]), angle=fmt(obj.data.angle))
        continue
    if obj.type != 'MESH' or obj.hide_render: continue
    mesh = obj.to_mesh()
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.calc_loop_triangles()
    mesh.calc_tangents() if mesh.uv_layers.active else None
    for mi in sorted({t.material_index for t in mesh.loop_triangles}):
        tris = [t for t in mesh.loop_triangles if t.material_index == mi]
        if not tris: continue
        vertices, normals, uv, tangents, signs = [], [], [], [], []
        normal_matrix = instance.matrix_world.to_3x3().inverted().transposed()
        for tri in tris:
            for vi, li in zip(tri.vertices, tri.loops):
                vertices.extend(instance.matrix_world @ mesh.vertices[vi].co)
                normals.extend((normal_matrix @ mesh.corner_normals[li].vector).normalized())
                if mesh.uv_layers.active:
                    uv.extend(mesh.uv_layers.active.data[li].uv)
                    tangents.extend((instance.matrix_world.to_3x3() @ mesh.loops[li].tangent).normalized())
                    signs.append(mesh.loops[li].bitangent_sign)
        material = mesh.materials[mi]
        state = ET.SubElement(root, 'state', shader=bpy.data.materials[material.name]['xml_name'], interpolation='smooth')
        objxml = ET.SubElement(state, 'object', name=obj.name + '_' + str(mi))
        attrs = dict(P=fmt(vertices), N=fmt(normals), nverts=' '.join(['3'] * len(tris)), verts=' '.join(map(str, range(len(vertices) // 3))))
        if uv:
            attrs.update(UV=fmt(uv), tangent=fmt(tangents), tangent_sign=fmt(signs))
        ET.SubElement(objxml, 'mesh', **attrs)
        triangles += len(tris)
    objects += 1
    obj.to_mesh_clear()
ET.ElementTree(root).write(OUT / 'sponza.xml', encoding='utf-8', xml_declaration=True)
(OUT / 'export.json').write_text(json.dumps(dict(source=bpy.data.filepath, objects=objects, triangles=triangles, materials=len(bpy.data.materials), textures=images), indent=2))
print('EXPORTED', objects, triangles, len(images), flush=True)
