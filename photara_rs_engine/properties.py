# SPDX-License-Identifier: Apache-2.0

import bpy


class PhotaraSceneSettings(bpy.types.PropertyGroup):
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

    adaptive_sampling: bpy.props.BoolProperty(
        name="Adaptive Sampling",
        description="Stop sampling pixels that converged early",
        default=True,
    )  # type: ignore[attr-defined]
