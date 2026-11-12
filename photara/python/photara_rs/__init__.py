"""Photara: an OptiX-first path tracer for Python and sim2real pipelines.

The public API is a small Blender-style scene graph::

    import numpy as np
    import photara_rs as pr

    scene = pr.Scene(1280, 720, samples=64)
    material = scene.materials.new("red", base_color=(0.8, 0.2, 0.1))
    mesh = scene.meshes.new("sphere", vertices, triangles, material=material)
    ball = scene.objects.new("ball", mesh, location=(0, 1, 0))
    ball.object_id = 7
    scene.camera.location = (4, 3, -6)
    scene.camera.target = (0, 1, 0)
    scene.lights.new_sun("sun", direction=(-0.4, -1.0, 0.2))
    frame = scene.render(seed=1)
    frame.beauty       # (H, W, 4) linear RGBA, top-down
    frame.save_npz("frame.npz")

A lower-level handle API is available as :class:`Session` for custom scene
builders and render farms. All colour data is linear float32; use
:func:`linear_to_srgb` and :func:`save_png` for previews.
"""

from __future__ import annotations

__version__ = "0.1.0"
try:  # Prefer the installed distribution metadata when available.
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _distribution_version

    try:
        __version__ = _distribution_version("photara-rs")
    except _PackageNotFoundError:
        pass
    del _PackageNotFoundError, _distribution_version
except ImportError:  # pragma: no cover - Python < 3.8 is not supported.
    pass

from ._errors import (
    PhotaraCancelledError,
    PhotaraClosedError,
    PhotaraError,
    PhotaraRemovedError,
    PhotaraRenderError,
)
from ._material import Material
from ._session import (
    RenderJob,
    RenderProgress,
    Session,
    linear_to_srgb,
    save_png,
)
from .importers import ImportResult, load_gltf, load_obj, parse_gltf, parse_obj
from .scene import Camera, Light, Mesh, Object, RenderResult, Scene, World

__all__ = [
    "Camera",
    "ImportResult",
    "Light",
    "Material",
    "Mesh",
    "Object",
    "PhotaraCancelledError",
    "PhotaraClosedError",
    "PhotaraError",
    "PhotaraRemovedError",
    "PhotaraRenderError",
    "RenderJob",
    "RenderProgress",
    "RenderResult",
    "Scene",
    "Session",
    "World",
    "__version__",
    "linear_to_srgb",
    "load_gltf",
    "load_obj",
    "parse_gltf",
    "parse_obj",
    "save_png",
]


def auto_denoiser():
    """Return the effective automatic denoiser for the selected OptiX device."""
    from ._backend import _core
    return {4: "OPENIMAGEDENOISE", 2: "OPTIX", 0: "NONE"}[_core.auto_denoiser()]

from .gsplat import GaussianSplats, load_gaussians, save_gaussian_ply
