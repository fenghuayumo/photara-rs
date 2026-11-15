"""Photara: an OptiX and Metal path tracer for Python and sim2real pipelines.

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

from pathlib import Path as _Path

_package_dir = _Path(__file__).resolve().parent
_repo_dir = _package_dir.parents[2]
_version_file = (_repo_dir / "VERSION" if (_repo_dir / "photara/CMakeLists.txt").is_file()
                 else _package_dir / "VERSION")
__version__ = _version_file.read_text(encoding="utf-8").strip()
del _Path, _package_dir, _repo_dir, _version_file

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
    """Return the effective automatic denoiser for the automatically selected render device."""
    from ._backend import _core
    return {4: "OPENIMAGEDENOISE", 2: "OPTIX", 0: "NONE"}[_core.auto_denoiser()]

from .gsplat import GaussianSplats, load_gaussians, save_gaussian_ply
