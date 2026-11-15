# SPDX-License-Identifier: Apache-2.0

import bpy
from bpy.app.handlers import persistent


def update_render_mode(self, context):
    scene = self.id_data
    if self.render_mode == 'REALTIME' and not scene.render.use_persistent_data:
        scene.render.use_persistent_data = True


@persistent
def begin_render_history(_scene=None):
    for scene in bpy.data.scenes:
        if scene.render.engine == 'PHOTARA_RS' and scene.photara_rs.render_mode == 'REALTIME':
            if not scene.render.use_persistent_data:
                scene.render.use_persistent_data = True
            scene.photara_rs.history_revision += 1


@persistent
def migrate_legacy_settings(_unused=None):
    """Copy saved renderer settings without claiming another add-on's property."""
    names = PhotaraSceneSettings.__annotations__
    temporary_pointer = not hasattr(bpy.types.Scene, 'photara')
    if temporary_pointer:
        # Blender 5.x stores RNA properties separately from custom ID properties.
        # Reattach the old name briefly to read saved RNA data, only if unclaimed.
        bpy.types.Scene.photara = bpy.props.PointerProperty(type=PhotaraSceneSettings)
    try:
        for scene in bpy.data.scenes:
            settings = scene.photara_rs
            if settings.keys():
                continue
            legacy = scene.photara
            values = {name: value for name, value in legacy.items() if name in names}
            # Also accept custom ID-property storage used by older Blender builds.
            raw = scene.get('photara')
            if raw is not None and hasattr(raw, 'items'):
                values = {**{k: v for k, v in raw.items() if k in names}, **values}
            for name, value in values.items():
                settings[name] = value
    finally:
        if temporary_pointer:
            del bpy.types.Scene.photara


DENOISERS = [('AUTO', 'Automatic', 'Prefer GPU OpenImageDenoise, then OptiX, matching Cycles'), ('DLSS', 'DLSS', 'NVIDIA DLSS denoising and optional upscaling'), ('OPTIX', 'OptiX', 'NVIDIA GPU denoiser'), ('OPENIMAGEDENOISE', 'OpenImageDenoise', 'Intel Open Image Denoise')]
DENOISE_PASSES = [('RGB', 'None', 'Color only'), ('RGB_ALBEDO', 'Albedo', 'Albedo guidance'), ('RGB_ALBEDO_NORMAL', 'Albedo and Normal', 'Albedo and normal guidance')]
PREFILTERS = [('NONE', 'None', 'Unfiltered guidance'), ('FAST', 'Fast', 'Filter jointly'), ('ACCURATE', 'Accurate', 'Prefilter guidance')]
QUALITIES = [('HIGH', 'High', 'Best quality'), ('BALANCED', 'Balanced', 'Balanced quality'), ('FAST', 'Fast', 'Fastest')]


class PhotaraSceneSettings(bpy.types.PropertyGroup):
    render_mode: bpy.props.EnumProperty(
        name="Render Mode", items=[('REFERENCE', 'Reference', 'Independent path-traced output frames'),
                                   ('REALTIME', 'RealTime', 'Low-SPP output with GPU temporal reconstruction')],
        default='REFERENCE',
        update=update_render_mode,
    )
    history_samples: bpy.props.IntProperty(
        name="History Frames", default=8, min=0, max=64,
        description="Maximum previous raw frames used by GPU reprojection; does not change Samples",
    )
    realtime_warmup_frames: bpy.props.IntProperty(
        name="Warm-up Frames", default=8, min=0, max=256,
        description="Additional internal frames before a still or the first animation frame; only the last is saved",
    )
    realtime_neeat: bpy.props.BoolProperty(name="NEE-AT", default=True)
    realtime_neeat_mode: bpy.props.EnumProperty(
        name="Light Sampling",
        items=[('PURE', 'NEE-AT Only', 'Global/local feedback without per-sample Light Tree evaluation'),
               ('HYBRID', 'NEE-AT + Light Tree', 'Mix 75% feedback and 25% Light Tree')],
        default='HYBRID',
    )
    history_revision: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    device: bpy.props.EnumProperty(
        name="Device",
        items=[('OPTIX', "GPU Compute", "Render on NVIDIA GPU with OptiX")],
        default='OPTIX',
    )

    viewport_denoise: bpy.props.BoolProperty(
        name="Denoise", description="Denoise viewport renders", default=True,
    )

    viewport_adaptive_sampling: bpy.props.BoolProperty(
        name="Adaptive Sampling", description="Stop sampling converged viewport pixels", default=True,
    )

    samples: bpy.props.IntProperty(
        name="Samples",
        description="Path tracing samples per pixel for final renders",
        default=128,
        min=1,
        max=1048576,
    )  # type: ignore[attr-defined]

    viewport_samples: bpy.props.IntProperty(
        name="Viewport Samples",
        description="Samples per pixel used for viewport renders",
        default=16,
        min=1,
        max=4096,
    )  # type: ignore[attr-defined]

    adaptive_threshold: bpy.props.FloatProperty(name="Noise Threshold", default=0.01, min=0.0, max=1.0, precision=4)
    adaptive_min_samples: bpy.props.IntProperty(name="Min Samples", default=0, min=0, max=1048576, description="Minimum samples before adaptive stopping; 0 selects automatically")
    time_limit: bpy.props.FloatProperty(name="Time Limit", default=0.0, min=0.0, subtype='TIME', unit='TIME', description="Render time limit in seconds; 0 disables the limit")
    viewport_adaptive_threshold: bpy.props.FloatProperty(name="Noise Threshold", default=0.1, min=0.0, max=1.0, precision=4)
    viewport_adaptive_min_samples: bpy.props.IntProperty(name="Min Samples", default=0, min=0, max=4096)
    viewport_denoise_start_sample: bpy.props.IntProperty(name="Start Sample", default=8, min=0, max=4096, description="Viewport sample count at which denoising starts; final output is always denoised")

    seed: bpy.props.IntProperty(
        name="Seed",
        description="Random seed; the same seed reproduces the same noise pattern",
        default=0,
        min=0,
        max=2147483647,
    )  # type: ignore[attr-defined]

    max_bounces: bpy.props.IntProperty(
        name="Max Bounces",
        description="Maximum number of light bounces",
        default=8,
        min=1,
        max=1024,
    )  # type: ignore[attr-defined]

    denoise: bpy.props.BoolProperty(
        name="Denoise",
        description="Denoise the combined pass with the device denoiser",
        default=True,
    )  # type: ignore[attr-defined]

    denoise_min_samples: bpy.props.IntProperty(
        name="Minimum SPP Before Denoise",
        description="Final images and animation frames: 0 follows Samples; otherwise render at least this many samples per pixel and disable adaptive early stopping",
        default=0, min=0, max=1048576,
    )

    adaptive_sampling: bpy.props.BoolProperty(
        name="Adaptive Sampling",
        description="Stop sampling pixels that converged early",
        default=True,
    )  # type: ignore[attr-defined]

    # Disabled capability indicators until cross-frame history is implemented.
    # Never save these as operational renderer settings.
    temporal_denoising_unavailable: bpy.props.BoolProperty(
        name="Temporal Denoising", default=False, options={'SKIP_SAVE'},
        description="Not yet supported by Photara: requires previous-frame output and motion vectors",
    )
    animation_history_unavailable: bpy.props.BoolProperty(
        name="Keep Animation History", default=False, options={'SKIP_SAVE'},
        description="Not yet supported by Photara: final render sessions are recreated per frame",
    )

    denoiser: bpy.props.EnumProperty(name="Denoiser", items=DENOISERS, default="AUTO")

    denoise_passes: bpy.props.EnumProperty(name="Passes", items=DENOISE_PASSES, default="RGB_ALBEDO_NORMAL")

    denoise_prefilter: bpy.props.EnumProperty(name="Prefilter", items=PREFILTERS, default="ACCURATE")

    denoise_quality: bpy.props.EnumProperty(name="Quality", items=QUALITIES, default="HIGH")

    denoise_use_gpu: bpy.props.BoolProperty(name="Use GPU", default=True)

    viewport_denoiser: bpy.props.EnumProperty(name="Denoiser", items=DENOISERS, default="AUTO")

    viewport_denoise_passes: bpy.props.EnumProperty(name="Passes", items=DENOISE_PASSES, default="RGB_ALBEDO_NORMAL")

    viewport_denoise_prefilter: bpy.props.EnumProperty(name="Prefilter", items=PREFILTERS, default="ACCURATE")

    viewport_denoise_quality: bpy.props.EnumProperty(name="Quality", items=QUALITIES, default="HIGH")

    viewport_denoise_use_gpu: bpy.props.BoolProperty(name="Use GPU", default=True)

    film_exposure: bpy.props.FloatProperty(name="Exposure", default=1.0, min=0.0, max=10.0)
    film_transparent_glass: bpy.props.BoolProperty(name="Transparent Glass", default=False)
    film_transparent_roughness: bpy.props.FloatProperty(name="Roughness Threshold", default=0.1, min=0.0, max=1.0)
    pixel_filter_type: bpy.props.EnumProperty(name="Type", items=[('BOX', 'Box', 'Box filter'), ('GAUSSIAN', 'Gaussian', 'Gaussian filter'), ('BLACKMAN_HARRIS', 'Blackman-Harris', 'Blackman-Harris filter')], default='BLACKMAN_HARRIS')
    filter_width: bpy.props.FloatProperty(name="Width", default=1.5, min=0.01, max=10.0)

    denoise_upscale_quality: bpy.props.EnumProperty(name="DLSS Upscaling", items=[('NONE', 'Native Resolution', 'Denoise at full resolution'), ('QUALITY', 'Quality', '1.5x upscale'), ('BALANCED', 'Balanced', '1.724x upscale'), ('PERFORMANCE', 'Performance', '2x upscale'), ('ULTRA_PERFORMANCE', 'Ultra Performance', '3x upscale')], default='NONE')
    viewport_denoise_upscale_quality: bpy.props.EnumProperty(name="DLSS Upscaling", items=[('NONE', 'Native Resolution', 'Denoise at full resolution'), ('QUALITY', 'Quality', '1.5x upscale'), ('BALANCED', 'Balanced', '1.724x upscale'), ('PERFORMANCE', 'Performance', '2x upscale'), ('ULTRA_PERFORMANCE', 'Ultra Performance', '3x upscale')], default='NONE')
