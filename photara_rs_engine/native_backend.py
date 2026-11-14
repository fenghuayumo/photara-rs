# SPDX-License-Identifier: Apache-2.0
"""Blender-native synchronization backed by Photara's independently built core.

Shader nodes, images, attributes and dependency-graph updates are handled by
the complete vendored BlenderSync. This adapter only maps render controls.
"""
from __future__ import annotations

import importlib
import os

_module = None
_registered = False
_configuring = set()
_last_configuration = {}


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
        # Blender otherwise destroys the renderer and its GPU context per frame.
        scene.render.use_persistent_data = True
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
        photara_neeat=settings.realtime_neeat, photara_history_revision=settings.history_revision,
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
    """Conservatively invalidate history for edits beyond camera animation."""
    scene = depsgraph.scene_eval.original
    camera = scene.camera
    for update in depsgraph.updates:
        data = update.id.original
        if data == camera or (camera and data == camera.data):
            continue
        if data.bl_rna.identifier in {'Scene', 'Collection', 'Action'}:
            continue
        scene.photara_rs.history_revision += 1
        break
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
    configure(context.scene)
    if not engine.session:
        view_update(engine, context, depsgraph)
    native_engine.view_draw(engine, depsgraph, context.region, context.space_data,
                            context.region_data)
