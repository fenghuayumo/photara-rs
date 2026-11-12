# SPDX-License-Identifier: Apache-2.0

import bpy


DENOISERS = [('AUTO', 'Automatic', 'Prefer OptiX'), ('DLSS', 'DLSS Ray Reconstruction', 'NVIDIA DLSS denoising and optional upscaling'), ('OPTIX', 'OptiX', 'NVIDIA GPU denoiser'), ('OPENIMAGEDENOISE', 'OpenImageDenoise', 'Intel Open Image Denoise')]
DENOISE_PASSES = [('RGB', 'None', 'Color only'), ('RGB_ALBEDO', 'Albedo', 'Albedo guidance'), ('RGB_ALBEDO_NORMAL', 'Albedo and Normal', 'Albedo and normal guidance')]
PREFILTERS = [('NONE', 'None', 'Unfiltered guidance'), ('FAST', 'Fast', 'Filter jointly'), ('ACCURATE', 'Accurate', 'Prefilter guidance')]
QUALITIES = [('HIGH', 'High', 'Best quality'), ('BALANCED', 'Balanced', 'Balanced quality'), ('FAST', 'Fast', 'Fastest')]


class PhotaraSceneSettings(bpy.types.PropertyGroup):
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
