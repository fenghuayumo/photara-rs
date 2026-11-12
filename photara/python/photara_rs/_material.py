"""Material data-block used by both the Scene and low-level Session APIs."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._errors import PhotaraError
from ._math import as_float3
from ._texture import normalize_texture_config

_FIELD_NAMES = (
    "base_color",
    "metallic",
    "roughness",
    "ior",
    "transmission",
    "emission_color",
    "emission_strength",
)

_DEFAULTS = {
    "base_color": (0.8, 0.8, 0.8),
    "metallic": 0.0,
    "roughness": 0.5,
    "ior": 1.5,
    "transmission": 0.0,
    "emission_color": (1.0, 1.0, 1.0),
    "emission_strength": 0.0,
}


def _validate_color(value, name: str) -> tuple[float, float, float]:
    array = as_float3(value, name)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must not contain negative values")
    return float(array[0]), float(array[1]), float(array[2])


def _validate_scalar(
    value,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def validate_values(values: dict) -> dict:
    """Validate a complete material parameter dictionary."""
    missing = [name for name in _FIELD_NAMES if name not in values]
    extra = [name for name in values if name not in _FIELD_NAMES]
    if missing:
        raise TypeError(f"Missing material fields: {', '.join(missing)}")
    if extra:
        raise TypeError(f"Unknown material fields: {', '.join(extra)}")

    return {
        "base_color": _validate_color(values["base_color"], "base_color"),
        "metallic": _validate_scalar(values["metallic"], "metallic", 0.0, 1.0),
        "roughness": _validate_scalar(values["roughness"], "roughness", 0.0, 1.0),
        "ior": _validate_scalar(values["ior"], "ior", 1e-6),
        "transmission": _validate_scalar(values["transmission"], "transmission", 0.0, 1.0),
        "emission_color": _validate_color(values["emission_color"], "emission_color"),
        "emission_strength": _validate_scalar(
            values["emission_strength"], "emission_strength", 0.0
        ),
    }


class Material:
    """A Principled BSDF material.

    Materials created through :attr:`Scene.materials` are bound to a scene: any
    property change is pushed to the renderer immediately and is visible on the
    next render. A standalone Material can be passed to
    :meth:`Session.add_material`, or attached to a Scene through
    ``scene.materials.new``.
    """

    def __init__(
        self,
        base_color=(0.8, 0.8, 0.8),
        metallic: float = 0.0,
        roughness: float = 0.5,
        ior: float = 1.5,
        transmission: float = 0.0,
        emission_color=(1.0, 1.0, 1.0),
        emission_strength: float = 0.0,
        name: Optional[str] = None,
    ) -> None:
        self.name = str(name) if name is not None else "Material"
        self._session = None
        self._index = -1
        self._texture = None
        self._values = validate_values(
            {
                "base_color": base_color,
                "metallic": metallic,
                "roughness": roughness,
                "ior": ior,
                "transmission": transmission,
                "emission_color": emission_color,
                "emission_strength": emission_strength,
            }
        )

    # ------------------------------------------------------------------
    # Binding.
    # ------------------------------------------------------------------
    def _bind(self, session, index: int) -> None:
        self._session = session
        self._index = int(index)

    def _unbind(self) -> None:
        self._session = None
        self._index = -1

    @property
    def index(self) -> int:
        """Renderer material index, or -1 for an unattached material."""
        return self._index

    @property
    def bound(self) -> bool:
        return self._session is not None and self._index >= 0

    def _sync(self) -> None:
        if self.bound:
            self._session._update_material_values(self._index, self.to_dict())

    def _set_values(self, values: dict, sync: bool = False) -> None:
        self._values = validate_values(values)
        if sync:
            self._sync()

    def _set_texture_config(self, config: Optional[dict]) -> None:
        self._texture = None if config is None else dict(config)

    # ------------------------------------------------------------------
    # Properties.
    # ------------------------------------------------------------------
    @property
    def base_color(self) -> tuple[float, float, float]:
        return self._values["base_color"]

    @base_color.setter
    def base_color(self, value) -> None:
        self._values["base_color"] = _validate_color(value, "base_color")
        self._sync()

    @property
    def metallic(self) -> float:
        return self._values["metallic"]

    @metallic.setter
    def metallic(self, value) -> None:
        self._values["metallic"] = _validate_scalar(value, "metallic", 0.0, 1.0)
        self._sync()

    @property
    def roughness(self) -> float:
        return self._values["roughness"]

    @roughness.setter
    def roughness(self, value) -> None:
        self._values["roughness"] = _validate_scalar(value, "roughness", 0.0, 1.0)
        self._sync()

    @property
    def ior(self) -> float:
        return self._values["ior"]

    @ior.setter
    def ior(self, value) -> None:
        self._values["ior"] = _validate_scalar(value, "ior", 1e-6)
        self._sync()

    @property
    def transmission(self) -> float:
        return self._values["transmission"]

    @transmission.setter
    def transmission(self, value) -> None:
        self._values["transmission"] = _validate_scalar(value, "transmission", 0.0, 1.0)
        self._sync()

    @property
    def emission_color(self) -> tuple[float, float, float]:
        return self._values["emission_color"]

    @emission_color.setter
    def emission_color(self, value) -> None:
        self._values["emission_color"] = _validate_color(value, "emission_color")
        self._sync()

    @property
    def base_color_texture(self) -> Optional[dict]:
        """Base-color texture configuration, or None when no texture is used."""
        if self._texture is None:
            return None
        return {
            "path": self._texture["path"],
            "colorspace": self._texture["colorspace"],
            "interpolation": self._texture["interpolation"],
            "extension": self._texture["extension"],
        }

    @base_color_texture.setter
    def base_color_texture(self, value) -> None:
        if value is None:
            if self.bound:
                self._session.clear_material_base_color_texture(self._index)
            self._texture = None
            return
        if not self.bound:
            raise PhotaraError(
                "Material must be attached to a Scene or Session before assigning a texture"
            )
        config = normalize_texture_config(value)
        self._session.set_material_base_color_texture(self._index, value)
        self._texture = config

    def configure_alpha_clip(self, cutoff=0.5):
        """Clip base-color image alpha; None disables. OMM is automatic when supported."""
        if cutoff is None:
            cutoff = -1.0
        else:
            cutoff = _validate_scalar(cutoff, "alpha cutoff", 0.0, 1.0)
        if not self.bound:
            raise PhotaraError("Attach material to a scene before configuring alpha clipping")
        self._session.set_material_alpha_clip(self._index, cutoff)

    @property
    def emission_strength(self) -> float:
        return self._values["emission_strength"]

    @emission_strength.setter
    def emission_strength(self, value) -> None:
        self._values["emission_strength"] = _validate_scalar(
            value, "emission_strength", 0.0
        )
        self._sync()

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------
    def update(self, **kwargs) -> "Material":
        """Set multiple fields at once and push one update to the renderer."""
        unknown = [name for name in kwargs if name not in _FIELD_NAMES]
        if unknown:
            raise TypeError(f"Unknown material fields: {', '.join(unknown)}")
        values = self.to_dict()
        values.update(kwargs)
        self._values = validate_values(values)
        self._sync()
        return self

    def to_dict(self) -> dict:
        return {
            "base_color": tuple(self._values["base_color"]),
            "metallic": float(self._values["metallic"]),
            "roughness": float(self._values["roughness"]),
            "ior": float(self._values["ior"]),
            "transmission": float(self._values["transmission"]),
            "emission_color": tuple(self._values["emission_color"]),
            "emission_strength": float(self._values["emission_strength"]),
        }

    def _as_args(self):
        values = self._values
        return (
            np.asarray(values["base_color"], dtype=np.float32),
            float(values["metallic"]),
            float(values["roughness"]),
            float(values["ior"]),
            float(values["transmission"]),
            np.asarray(values["emission_color"], dtype=np.float32),
            float(values["emission_strength"]),
        )

    def copy(self, name: Optional[str] = None) -> "Material":
        """Return an unattached copy of this material."""
        values = self.to_dict()
        values["name"] = self.name if name is None else name
        return Material(**values)

    def __repr__(self) -> str:
        state = "bound" if self.bound else "detached"
        return (
            f"Material(name={self.name!r}, base_color={self.base_color}, "
            f"roughness={self.roughness}, {state})"
        )


__all__ = ["Material", "validate_values"]
