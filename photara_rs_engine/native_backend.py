# SPDX-License-Identifier: Apache-2.0
"""Blender-native synchronization backed by Photara's independently built core.

Shader nodes, images, attributes and dependency-graph updates are handled by
the complete vendored BlenderSync. This adapter only maps render controls.
"""
from __future__ import annotations

import importlib
import os
from array import array
import zlib

_module = None
_registered = False
_configuring = set()
_last_configuration = {}
_last_frame_state = {}


def available():
    global _module
    if os.environ.get('PHOTARA_LEGACY_BLENDER_SYNC') == '1':
        return False
    if _module is None:
        try:
            _module = importlib.import_module('_photara_native')
        except ImportError:
            return False
    return True


def register():
    global _registered
    if not available() or _registered:
        return
    import bpy
    from . import native_engine, native_properties
    bpy.utils.register_class(native_properties.PhotaraNativeRenderSettings)
    native_engine.init()
    _registered = True


def unregister():
    global _registered
    if not _registered:
        return
    import bpy
    from . import native_engine, native_properties
    native_engine.exit()
    bpy.utils.unregister_class(native_properties.PhotaraNativeRenderSettings)
    _registered = False
    _last_configuration.clear()
    _last_frame_state.clear()


def configure(scene):
    original = getattr(scene, 'original', None)
    scene = original if original is not None else scene
    key = scene.as_pointer()
    if key in _configuring:
        return
    _configuring.add(key)
    try:
        _configure(scene)
    finally:
        _configuring.discard(key)


def _configure(scene):
    """Keep Photara controls separate from the user's Cycles scene settings."""
    original = getattr(scene, 'original', None)
    scene = original if original is not None else scene
    native = scene.photara_native
    settings = scene.photara_rs
    realtime = settings.render_mode == 'REALTIME'
    if realtime:
        if not hasattr(_module, 'realtime_stats'):
            raise RuntimeError('RealTime output requires a rebuilt _photara_native module')
    if not native.get('_photara_initialized', False):
        # Preserve the complete scene-level schema, including controls not yet
        # exposed in Photara's compact panels. Material/object/world properties
        # are read directly by BlenderSync from the original datablocks.
        for prop in scene.cycles.bl_rna.properties:
            name = prop.identifier
            if prop.is_readonly or prop.type not in {'BOOLEAN', 'INT', 'FLOAT', 'STRING', 'ENUM'}:
                continue
            if hasattr(native, name):
                setattr(native, name, getattr(scene.cycles, name))
        native['_photara_initialized'] = True
    values = dict(
        photara_realtime=realtime, photara_history_samples=settings.history_samples,
        photara_warmup_frames=settings.realtime_warmup_frames,
        photara_neeat=settings.realtime_neeat,
        photara_neeat_tree_fraction=0.0 if settings.realtime_neeat_mode == 'PURE' else 0.25,
        photara_history_revision=settings.history_revision,
        device='GPU', use_pixel_jitter=True, samples=(settings.samples if realtime else max(settings.samples,
                                   settings.denoise_min_samples if settings.denoise else 0)),
        preview_samples=settings.viewport_samples, seed=settings.seed,
        max_bounces=settings.max_bounces, diffuse_bounces=settings.max_bounces,
        glossy_bounces=settings.max_bounces, transmission_bounces=settings.max_bounces,
        use_adaptive_sampling=not realtime and settings.adaptive_sampling and
                              not (settings.denoise and settings.denoise_min_samples),
        use_preview_adaptive_sampling=settings.viewport_adaptive_sampling,
        adaptive_threshold=settings.adaptive_threshold,
        adaptive_min_samples=settings.adaptive_min_samples,
        preview_adaptive_threshold=settings.viewport_adaptive_threshold,
        preview_adaptive_min_samples=settings.viewport_adaptive_min_samples,
        time_limit=settings.time_limit, use_denoising=settings.denoise,
        use_preview_denoising=settings.viewport_denoise,
        denoiser=(settings.denoiser if settings.denoiser != 'AUTO' else
                  'OPENIMAGEDENOISE' if _module.with_openimagedenoise else 'OPTIX'),
        preview_denoiser=settings.viewport_denoiser,
        denoising_input_passes=settings.denoise_passes,
        preview_denoising_input_passes=settings.viewport_denoise_passes,
        denoising_prefilter=settings.denoise_prefilter,
        preview_denoising_prefilter=settings.viewport_denoise_prefilter,
        denoising_quality=settings.denoise_quality,
        preview_denoising_quality=settings.viewport_denoise_quality,
        denoising_use_gpu=settings.denoise_use_gpu,
        preview_denoising_use_gpu=settings.viewport_denoise_use_gpu,
        preview_denoising_start_sample=settings.viewport_denoise_start_sample,
        denoising_upscale_quality=settings.denoise_upscale_quality,
        preview_denoising_upscale_quality=settings.viewport_denoise_upscale_quality,
        film_exposure=settings.film_exposure,
        film_transparent_glass=settings.film_transparent_glass,
        film_transparent_roughness=settings.film_transparent_roughness,
        pixel_filter_type=settings.pixel_filter_type, filter_width=settings.filter_width,
    )
    signature = tuple(values.items())
    key = scene.as_pointer()
    if _last_configuration.get(key) == signature:
        return
    for name, value in values.items():
        if getattr(native, name) != value:
            setattr(native, name, value)
    _last_configuration[key] = signature


def configure_frame(depsgraph):
    """Retain continuous geometry animation; invalidate edits and topology changes."""
    scene = depsgraph.scene_eval.original
    if scene.photara_rs.render_mode != 'REALTIME':
        _last_frame_state.pop(scene.as_pointer(), None)
        configure(scene)
        return
    # Connectivity and object/data identities catch topology, replacement and
    # membership changes without invalidating vertex/transform animation.
    connectivity = {}
    for obj in depsgraph.objects:
        if obj.type == 'MESH' and obj.data.as_pointer() not in connectivity:
            indices = array('i', [0]) * len(obj.data.loops)
            obj.data.loops.foreach_get('vertex_index', indices)
            connectivity[obj.data.as_pointer()] = zlib.crc32(indices.tobytes())
    topology = tuple(sorted(
        (obj.original.as_pointer(), obj.original.data.as_pointer() if obj.original.data else 0,
         len(obj.data.vertices) if obj.type == 'MESH' else 0,
         len(obj.data.polygons) if obj.type == 'MESH' else 0,
         connectivity.get(obj.data.as_pointer(), 0) if obj.data else 0,
         tuple(slot.material.original.as_pointer() if slot.material else 0
               for slot in obj.material_slots))
        for obj in depsgraph.objects))
    key = scene.as_pointer()
    frame = (scene.frame_current, scene.frame_subframe)
    previous = _last_frame_state.get(key)
    continuous = previous is not None and previous[0] == (frame[0] - 1, 0.0) and frame[1] == 0.0
    same_topology = previous is not None and previous[1] == topology
    _last_frame_state[key] = (frame, topology)
    camera = scene.camera
    invalidate = previous is not None and not same_topology
    for update in depsgraph.updates:
        data = update.id.original
        if data == camera or (camera and data == camera.data):
            continue
        # Compositing happens after renderer history and cannot invalidate the
        # raw radiance cache. Blender tags its tree on every animated output.
        if data.bl_rna.identifier in {'Scene', 'Collection', 'Action', 'CompositorNodeTree'}:
            continue
        if data.bl_rna.identifier == 'Object' and data.type == 'LIGHT':
            invalidate = True
            continue
        if continuous and same_topology and data.bl_rna.identifier in {
                'Object', 'Mesh', 'Armature', 'Key', 'Curve', 'Curves', 'PointCloud'}:
            continue
        invalidate = True
    if invalidate:
        scene.photara_rs.history_revision += 1
    configure(scene)


def view_update(engine, context, depsgraph):
    from . import native_engine
    configure(context.scene)
    if not engine.session:
        native_engine.create(engine, context.blend_data, context.region,
                             context.space_data, context.region_data)
    native_engine.reset(engine, context.blend_data, depsgraph)
    native_engine.sync(engine, depsgraph, context.blend_data)


def view_draw(engine, context, depsgraph):
    from . import native_engine
    if not engine.session:
        # Draw callbacks cannot write IDs. Ask Blender for a writable update
        # instead of configuring properties or creating/resetting sessions here.
        engine.tag_update()
        return
    native_engine.view_draw(engine, depsgraph, context.region, context.space_data,
                            context.region_data)
