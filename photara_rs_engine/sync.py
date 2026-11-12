# SPDX-License-Identifier: Apache-2.0
"""Synchronize a Blender dependency graph into a photara_rs scene.

The sync keeps one :class:`photara_rs.Scene` alive across dependency graph
updates and applies incremental changes (transforms, deformed meshes, edited
materials and lights) wherever possible, so viewport renders do not rebuild
the whole renderer session on every edit.

Coordinate conventions
----------------------
World-space data is passed through unchanged; only local conventions differ:

* Mesh objects: ``matrix_world`` is passed as the photara object transform.
* Lights: Blender lights emit along local -Z, which matches photara light
  matrices, so ``matrix_world`` is passed through.
* Camera: Blender cameras look along local -Z while photara cameras look
  along local +Z, so the camera matrix is post-multiplied by a Z flip (the
  same conversion Cycles applies in ``blender_camera_matrix``).
"""

from __future__ import annotations

import math

import bpy
import numpy as np


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def _socket_color(node, name, default):
    try:
        socket = node.inputs[name]
    except KeyError:
        return default
    if socket is None:
        return default
    value = socket.default_value
    try:
        return tuple(float(x) for x in value[:3])
    except TypeError:
        return default


def _node_input(node, name):
    """Look up a node input by name without collection `.get()` (Blender 5.x
    only supports `.get()` on bpy.data collections)."""
    try:
        return node.inputs[name]
    except KeyError:
        return None


def _socket_scalar(node, names, default):
    for name in names:
        try:
            socket = node.inputs[name]
        except KeyError:
            continue
        if socket is not None:
            return float(socket.default_value)
    return default


def sync_material(state, blender_material):
    """Create or update the photara material for a Blender material."""
    key = blender_material.original.as_pointer()
    material = state.materials.get(key)
    if material is None:
        material = state.scene.materials.new(blender_material.name)
        state.materials[key] = material

    base_color = (0.8, 0.8, 0.8)
    metallic = 0.0
    roughness = 0.5
    ior = 1.45
    transmission = 0.0
    emission_color = (0.0, 0.0, 0.0)
    emission_strength = 0.0

    node_tree = getattr(blender_material, "node_tree", None)
    if node_tree is not None:
        principled = None
        for node in node_tree.nodes:
            # `node.type` is stable across Blender versions; `bl_idname`
            # changed shape historically (BSDF_PRINCIPLED vs
            # ShaderNodeBsdfPrincipled).
            if getattr(node, "type", "") == 'BSDF_PRINCIPLED':
                principled = node
                break
        if principled is not None:
            base_color = _socket_color(principled, "Base Color", base_color)
            metallic = _socket_scalar(principled, ("Metallic",), metallic)
            roughness = _socket_scalar(principled, ("Roughness",), roughness)
            ior = _socket_scalar(principled, ("IOR",), ior)
            # Blender 4.0 renamed several Principled BSDF sockets.
            transmission = _socket_scalar(
                principled, ("Transmission Weight", "Transmission"), transmission
            )
            emission_color = _socket_color(
                principled, ("Emission Color", "Emission"), emission_color
            )
            emission_strength = _socket_scalar(
                principled, ("Emission Strength",), emission_strength
            )

    material.update(
        base_color=base_color,
        metallic=max(0.0, min(1.0, metallic)),
        roughness=max(0.0, min(1.0, roughness)),
        ior=max(1e-6, ior),
        transmission=max(0.0, min(1.0, transmission)),
        emission_color=emission_color,
        emission_strength=max(0.0, emission_strength),
    )
    return material


# ---------------------------------------------------------------------------
# Mesh extraction
# ---------------------------------------------------------------------------


def mesh_triangles(mesh):
    """Return ``(positions, normals, uvs, indices)`` as corner arrays.

    Corner arrays duplicate every triangle vertex, which keeps UV seams and
    split normals exact.  ``positions``/``uvs`` have shapes ``(N, 3)`` and
    ``(N, 2)``; ``indices`` indexes corners directly.
    """
    mesh.calc_loop_triangles()

    num_triangles = len(mesh.loop_triangles)
    if num_triangles == 0:
        empty3 = np.zeros((0, 3), dtype=np.float32)
        empty2 = np.zeros((0, 2), dtype=np.float32)
        return empty3, empty3.copy(), empty2, np.zeros(0, dtype=np.int32)

    num_corners = num_triangles * 3
    vertex_count = len(mesh.vertices)

    verts = np.empty(vertex_count * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3)

    corner_verts = np.empty(num_corners, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", corner_verts)
    corner_loops = np.empty(num_corners, dtype=np.int64)
    mesh.loop_triangles.foreach_get("loops", corner_loops)

    positions = np.ascontiguousarray(verts[corner_verts], dtype=np.float32)

    # Split normals: corner_normals in Blender 4.1+, loop normals before that,
    # vertex normals as a last resort.
    normals = None
    try:
        loop_normals = np.empty(len(mesh.loops) * 3, dtype=np.float32)
        # Blender 5.x exposes `vector` on MeshNormalValue; 4.x used `normal`.
        try:
            mesh.corner_normals.foreach_get("vector", loop_normals)
        except AttributeError:
            mesh.corner_normals.foreach_get("normal", loop_normals)
        normals = np.ascontiguousarray(
            loop_normals.reshape(-1, 3)[corner_loops], dtype=np.float32
        )
    except Exception:
        try:
            mesh.calc_normals_split()
            loop_normals = np.empty(len(mesh.loops) * 3, dtype=np.float32)
            mesh.loops.foreach_get("normal", loop_normals)
            normals = np.ascontiguousarray(
                loop_normals.reshape(-1, 3)[corner_loops], dtype=np.float32
            )
        except Exception:
            vertex_normals = np.empty(vertex_count * 3, dtype=np.float32)
            mesh.vertices.foreach_get("normal", vertex_normals)
            normals = np.ascontiguousarray(
                vertex_normals.reshape(-1, 3)[corner_verts], dtype=np.float32
            )

    # Active UV layer (per loop corner).
    uvs = np.zeros((num_corners, 2), dtype=np.float32)
    uv_layer = mesh.uv_layers.active
    if uv_layer is not None:
        try:
            loop_uvs = np.empty(len(mesh.loops) * 2, dtype=np.float32)
            uv_layer.data.foreach_get("uv", loop_uvs)
            uvs = np.ascontiguousarray(
                loop_uvs.reshape(-1, 2)[corner_loops], dtype=np.float32
            )
        except Exception:
            pass

    indices = np.arange(num_corners, dtype=np.int32).reshape(-1, 3)
    return positions, normals, uvs, indices


# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------


def _light_axes(matrix_world):
    import mathutils

    rotation = matrix_world.to_3x3()
    forward = (rotation @ mathutils.Vector((0.0, 0.0, -1.0))).normalized()
    up = rotation @ mathutils.Vector((0.0, 1.0, 0.0))
    if abs(up.dot(forward)) > 0.999:
        up = rotation @ mathutils.Vector((1.0, 0.0, 0.0))
    return matrix_world.translation.copy(), forward, up


def sync_light(state, blender_light, matrix_world, key=None):
    if key is None:
        key = blender_light.original.as_pointer()
    light = state.lights.get(key)
    name = blender_light.name
    color = tuple(float(c) for c in blender_light.color)
    energy = float(blender_light.energy)
    light_type = blender_light.type

    position, forward, up = _light_axes(matrix_world)
    target = position + forward

    if light is not None and light.light_type != light_type:
        state.scene.objects.remove(light)
        light = None

    if light is None:
        scene = state.scene
        if light_type == 'AREA':
            size_x = float(blender_light.size)
            if blender_light.shape == 'RECT':
                size_y = float(blender_light.size_y)
            else:
                size_y = float(blender_light.size)
            light = scene.lights.new_area(
                name,
                position=position,
                target=target,
                size=(size_x, size_y),
                color=color,
                strength=energy,
                up=up,
            )
        elif light_type == 'SUN':
            light = scene.lights.new_sun(
                name,
                direction=forward,
                color=color,
                strength=energy,
                angle=float(blender_light.angle),
            )
        elif light_type == 'SPOT':
            light = scene.lights.new_spot(
                name,
                position=position,
                direction=forward,
                color=color,
                strength=energy,
                radius=float(blender_light.shadow_soft_size),
                angle=math.degrees(float(blender_light.spot_size)),
                smooth=float(blender_light.spot_blend),
            )
        else:
            light = scene.lights.new_point(
                name,
                position=position,
                color=color,
                strength=energy,
                radius=float(getattr(blender_light, "shadow_soft_size", 0.0)),
            )
        state.lights[key] = light
        return light

    # All photara light matrices follow Blender's convention (emit along
    # local -Z), so the world matrix can be passed through directly.
    light.matrix_world = np.asarray(matrix_world, dtype=np.float32)
    light.color = color
    light.strength = energy
    if light_type == 'AREA':
        light.size = (float(blender_light.size), float(blender_light.size))
        if blender_light.shape == 'RECT':
            light.size = (float(blender_light.size), float(blender_light.size_y))
    elif light_type == 'SUN':
        light.angle = float(blender_light.angle)
    elif light_type == 'SPOT':
        light.spot_angle = math.degrees(float(blender_light.spot_size))
        light.smooth = float(blender_light.spot_blend)
        light.radius = float(blender_light.shadow_soft_size)
    else:
        light.radius = float(getattr(blender_light, "shadow_soft_size", 0.0))
    return light


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


def sync_world(state, blender_world):
    world = state.scene.world
    color = (0.05, 0.05, 0.05)
    strength = 1.0
    env_path = None

    node = None
    if blender_world is not None and blender_world.use_nodes:
        node_tree = blender_world.node_tree
        if node_tree is not None:
            output = None
            for candidate in node_tree.nodes:
                if getattr(candidate, "type", "") == 'OUTPUT_WORLD':
                    output = candidate
                    if candidate.is_active_output:
                        break
            if output is not None:
                surface = _node_input(output, "Surface")
                if surface is not None and surface.is_linked:
                    node = surface.links[0].from_socket.node

    if node is not None and getattr(node, "type", "") == 'BACKGROUND':
        color_socket = _node_input(node, "Color")
        strength_socket = _node_input(node, "Strength")
        if color_socket is not None:
            if color_socket.is_linked:
                source = color_socket.links[0].from_socket.node
                if getattr(source, "type", "") == 'TEX_ENVIRONMENT':
                    image = source.image
                    if image is not None and image.source == 'FILE' and image.filepath:
                        import bpy

                        env_path = bpy.path.abspath(image.filepath)
                        color = (1.0, 1.0, 1.0)
            else:
                color = tuple(float(x) for x in color_socket.default_value[:3])
        if strength_socket is not None:
            strength = float(strength_socket.default_value)
    elif blender_world is not None and not blender_world.use_nodes:
        color = tuple(float(c) for c in blender_world.color)

    values = (color, strength, env_path)
    if state._world_values == values:
        return
    state._world_values = values
    world.color = color
    world.strength = strength
    if env_path and str(env_path).lower().endswith((".hdr", ".exr", ".pic", ".tga")):
        try:
            world.environment_texture = {"path": env_path}
            return
        except Exception:
            pass
    world.environment_texture = None


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def camera_matrix_photara(matrix_world):
    """Convert a Blender camera matrix to a photara camera matrix."""
    import mathutils

    return matrix_world @ mathutils.Matrix.Diagonal((1.0, 1.0, -1.0, 1.0))


def camera_fov_degrees(camera, aspect):
    """Photara's smaller-axis field of view, honouring Blender sensor fit."""
    sensor_width = float(camera.sensor_width)
    lens = max(1e-4, float(camera.lens))
    fit = camera.sensor_fit
    if fit == 'HORIZONTAL' or (fit == 'AUTO' and aspect >= 1.0):
        fov = 2.0 * math.atan(sensor_width / (2.0 * lens))
    else:
        sensor_height = float(camera.sensor_height) if fit == 'VERTICAL' else sensor_width
        fov_v = 2.0 * math.atan(sensor_height / (2.0 * lens))
        fov = 2.0 * math.atan(math.tan(fov_v / 2.0) * aspect)
    # Photara's auto viewplane defines FOV along the smaller image axis.
    if aspect > 1.0:
        fov = 2.0 * math.atan(math.tan(fov / 2.0) / aspect)
    return math.degrees(fov)


def fov_from_projection_degrees(projection, aspect):
    """Photara FOV in degrees from a viewport projection matrix.

    `RegionView3D.window_matrix` is the projection Blender actually draws
    with, so this is robust against the `lens` attribute moving between
    RNA types (RegionView3D in Blender <=4.x, SpaceView3D in 5.x) and
    against viewport zoom in camera view.
    """
    f = float(projection[1][1])  # 1 / tan(fov_vertical / 2)
    if not (f > 1e-6) or not math.isfinite(f):
        return None
    fov_v = 2.0 * math.atan(1.0 / f)
    fov = fov_v if aspect >= 1.0 else 2.0 * math.atan(aspect * math.tan(fov_v / 2.0))
    return math.degrees(fov)


# ---------------------------------------------------------------------------
# Incremental scene state
# ---------------------------------------------------------------------------


class SceneSyncState:
    """Owns a photara Scene and applies incremental depsgraph updates."""

    def __init__(self, scene):
        self.scene = scene
        self.meshes = {}       # blender mesh pointer -> photara Mesh
        self.mesh_counts = {}  # blender mesh pointer -> (num corners, num triangles)
        self.objects = {}      # blender object pointer -> photara Object
        self.object_mesh_keys = {}  # blender object pointer -> blender mesh pointer
        self.lights = {}       # blender light pointer -> photara Light
        self.materials = {}    # blender material pointer -> photara Material
        self.object_matrices = {}  # blender object pointer -> flattened matrix
        self._full_sync_done = False
        self._pending_geometry = set()
        self._pending_objects = set()
        self._pending_lights = set()
        self._pending_materials = set()
        self._pending_world = False
        self._world_values = None

    def close(self):
        try:
            self.scene.close()
        except Exception:
            pass

    def _first_material(self, blender_object, mesh):
        if blender_object.material_slots:
            slot = blender_object.material_slots[0]
            if slot.material is not None:
                return slot.material
        if mesh.materials and mesh.materials[0] is not None:
            return mesh.materials[0]
        return None

    def _sync_mesh_data(self, mesh_eval, blender_object):
        # Evaluated meshes may be temporary; key by their owning object so
        # linked meshes with different modifiers cannot overwrite each other.
        key = blender_object.original.as_pointer()
        positions, normals, uvs, indices = mesh_triangles(mesh_eval)
        if len(indices) == 0:
            return None

        blender_material = self._first_material(blender_object, mesh_eval)
        material = None
        if blender_material is not None:
            material = sync_material(self, blender_material)

        mesh = self.meshes.get(key)
        if mesh is None:
            mesh = self.scene.meshes.new(
                mesh_eval.name,
                positions,
                indices,
                normals=normals,
                uvs=uvs,
                material=material,
            )
            self.meshes[key] = mesh
            self.mesh_counts[key] = (len(positions), len(indices))
            mesh._blender_key = key
            return mesh

        old_corners, old_tris = self.mesh_counts.get(key, (-1, -1))
        if len(positions) != old_corners or len(indices) != old_tris:
            mesh.update_topology(
                positions,
                indices,
                normals=normals,
                uvs=uvs,
                material_index=-1 if material is None else material.index,
            )
            self.mesh_counts[key] = (len(positions), len(indices))
        else:
            mesh.update_positions(positions, normals=normals, uvs=uvs)
        if mesh.material is not material:
            mesh.material = material
        return mesh

    def _collect_updates(self, depsgraph):
        """Collect the subset of IDs that actually changed this update.

        ``depsgraph.updates`` is the same mechanism Cycles' BlenderSync uses
        to avoid touching unchanged data.  IDs are keyed by the pointer of
        their *original* datablock so they match the instance keys below.
        Material parameter edits are refreshed directly here: they only push
        a few shader uniforms and must not trigger geometry work.
        """
        dirty_geometry = set()
        dirty_objects = set()
        dirty_lights = set()
        dirty_materials = set()
        world_dirty = False

        for update in depsgraph.updates:
            updated_id = update.id
            original = updated_id.original or updated_id
            key = original.as_pointer()
            if isinstance(updated_id, bpy.types.Mesh):
                dirty_geometry.add(key)
            elif isinstance(updated_id, bpy.types.Light):
                dirty_lights.add(key)
            elif isinstance(updated_id, bpy.types.Material):
                dirty_materials.add(key)
            elif isinstance(updated_id, bpy.types.Object):
                # Modifier-stack changes surface as object updates carrying
                # the geometry flag; the mesh datablock itself is unchanged.
                dirty_objects.add(key)
                if update.is_updated_geometry:
                    dirty_geometry.add(key)
            elif isinstance(updated_id, (bpy.types.Scene, bpy.types.World)):
                world_dirty = True

        return dirty_geometry, dirty_objects, dirty_lights, world_dirty, dirty_materials

    def queue_updates(self, depsgraph):
        # Snapshot dirty IDs only: do not mutate a running native scene. Keep
        # the union because depsgraph.updates can be replaced before we render.
        geometry, objects, lights, world, materials = self._collect_updates(depsgraph)
        self._pending_geometry.update(geometry)
        self._pending_objects.update(objects)
        self._pending_lights.update(lights)
        self._pending_materials.update(materials)
        self._pending_world |= world

    def sync(self, depsgraph, sync_background=True):
        self.queue_updates(depsgraph)
        dirty_geometry, self._pending_geometry = self._pending_geometry, set()
        dirty_objects, self._pending_objects = self._pending_objects, set()
        dirty_lights, self._pending_lights = self._pending_lights, set()
        dirty_materials, self._pending_materials = self._pending_materials, set()
        world_dirty, self._pending_world = self._pending_world, False
        for material in bpy.data.materials:
            if material.as_pointer() in dirty_materials and material.as_pointer() in self.materials:
                sync_material(self, material)
        first_sync = not self._full_sync_done
        if sync_background and (first_sync or world_dirty):
            sync_world(self, depsgraph.scene_eval.world)

        seen_objects = set()
        seen_lights = set()

        # One cheap Python pass over instances detects additions/removals and
        # applies only the flagged updates.  Unchanged instances cost one dict
        # lookup and a matrix comparison each.
        for instance in depsgraph.object_instances:
            blender_object = instance.object
            original = blender_object.original
            if original is None:
                continue
            object_key = original.as_pointer()
            key = object_key
            if instance.is_instance:
                parent = instance.parent
                key = (object_key, parent.original.as_pointer() if parent else 0,
                       tuple(instance.persistent_id))
            if not instance.show_self:
                continue
            matrix_world = instance.matrix_world

            if blender_object.type == 'MESH':
                matrix_flat = np.asarray(matrix_world, dtype=np.float32).ravel().tolist()
                mesh_key = self.object_mesh_keys.get(key)

                photara_object = self.objects.get(key)
                if photara_object is None:
                    mesh_eval = blender_object.to_mesh()
                    if mesh_eval is None:
                        continue
                    try:
                        photara_mesh = self._sync_mesh_data(mesh_eval, blender_object)
                    finally:
                        blender_object.to_mesh_clear()
                    if photara_mesh is None:
                        continue
                    photara_object = self.scene.objects.new(
                        original.name,
                        photara_mesh,
                        transform=np.asarray(matrix_world, dtype=np.float32),
                        object_id=int(original.pass_index),
                    )
                    self.objects[key] = photara_object
                    blender_mesh_key = getattr(photara_mesh, "_blender_key", None)
                    if blender_mesh_key is not None:
                        self.object_mesh_keys[key] = blender_mesh_key
                    self.object_matrices[key] = matrix_flat
                    seen_objects.add(key)
                    continue

                transform_dirty = (
                    first_sync
                    or object_key in dirty_objects
                    or self.object_matrices.get(key) != matrix_flat
                )
                if transform_dirty:
                    photara_object.matrix_world = np.asarray(
                        matrix_world, dtype=np.float32
                    )
                    self.object_matrices[key] = matrix_flat
                    if int(original.pass_index) != photara_object.object_id:
                        photara_object.object_id = int(original.pass_index)
                    blender_material = self._first_material(blender_object, blender_object.data)
                    material = None
                    if blender_material is not None:
                        material = self.materials.get(blender_material.original.as_pointer())
                        if material is None:
                            material = sync_material(self, blender_material)
                    if photara_object.material is not material:
                        photara_object.material = material

                geometry_dirty = (
                    first_sync
                    or mesh_key is None
                    or mesh_key in dirty_geometry
                    or object_key in dirty_geometry
                    or (blender_object.data.original.as_pointer() in dirty_geometry)
                )
                if geometry_dirty:
                    mesh_eval = blender_object.to_mesh()
                    if mesh_eval is not None:
                        try:
                            new_mesh = self._sync_mesh_data(mesh_eval, blender_object)
                        finally:
                            blender_object.to_mesh_clear()
                        if new_mesh is None:
                            continue
                        current_mesh = getattr(photara_object, "mesh", None)
                        if current_mesh is not new_mesh:
                            # The object now uses a different mesh datablock;
                            # photara objects cannot re-link geometry, so
                            # rebuild the instance.
                            try:
                                self.scene.objects.remove(photara_object)
                            except Exception:
                                pass
                            self.objects.pop(key, None)
                            recreated = self.scene.objects.new(
                                original.name,
                                new_mesh,
                                transform=np.asarray(matrix_world, dtype=np.float32),
                                object_id=int(original.pass_index),
                            )
                            self.objects[key] = recreated
                            blender_mesh_key = getattr(new_mesh, "_blender_key", None)
                            if blender_mesh_key is not None:
                                self.object_mesh_keys[key] = blender_mesh_key
                seen_objects.add(key)

            elif blender_object.type == 'LIGHT':
                light_data = blender_object.data
                light_original = light_data.original or light_data
                light_key = key
                light_dirty = (
                    first_sync or light_original.as_pointer() in dirty_lights
                    or object_key in dirty_objects or light_key not in self.lights
                )
                if not light_dirty and self.lights.get(light_key) is not None:
                    # Still honour transform-only moves of the light object.
                    light_matrix = self.object_matrices.get(key)
                    current = np.asarray(matrix_world, dtype=np.float32).ravel().tolist()
                    light_dirty = light_matrix != current
                if light_dirty:
                    sync_light(self, light_data, matrix_world, key=light_key)
                    self.object_matrices[key] = np.asarray(
                        matrix_world, dtype=np.float32
                    ).ravel().tolist()
                seen_lights.add(light_key)

        self._full_sync_done = True

        for key in list(self.objects.keys()):
            if key not in seen_objects:
                obj = self.objects.pop(key)
                self.object_matrices.pop(key, None)
                self.object_mesh_keys.pop(key, None)
                try:
                    self.scene.objects.remove(obj)
                except Exception:
                    pass

        for light_key in list(self.lights.keys()):
            if light_key not in seen_lights:
                light = self.lights.pop(light_key)
                try:
                    self.scene.objects.remove(light)
                except Exception:
                    pass

        # Drop mesh data that is no longer referenced by any object.
        referenced_mesh_keys = set(self.object_mesh_keys.values())
        for mesh_key in list(self.meshes.keys()):
            if mesh_key not in referenced_mesh_keys:
                mesh = self.meshes.pop(mesh_key, None)
                if mesh is not None and mesh.users == 0:
                    self.scene.meshes.remove(mesh)
                self.mesh_counts.pop(mesh_key, None)

    def sync_camera(self, blender_camera_object, aspect=1.0):
        camera = blender_camera_object.data
        self.scene.camera.matrix_world = np.asarray(
            camera_matrix_photara(blender_camera_object.matrix_world), dtype=np.float32
        )
        self.scene.camera.fov = camera_fov_degrees(camera, aspect)
        self.scene.camera.near_clip = max(1e-4, float(camera.clip_start))
        self.scene.camera.far_clip = max(
            float(camera.clip_end), float(camera.clip_start) + 1e-3
        )

    def sync_camera_view(self, region_data, width=1, height=1):
        """Camera from viewport navigation (no scene camera object)."""
        matrix_world = region_data.view_matrix.inverted()
        self.scene.camera.matrix_world = np.asarray(
            camera_matrix_photara(matrix_world), dtype=np.float32
        )
        aspect = max(1e-6, float(width) / max(1.0, float(height)))
        fov = fov_from_projection_degrees(region_data.window_matrix, aspect)
        if fov is None or not (fov > 0.1):
            fov = 50.0
        self.scene.camera.fov = fov
