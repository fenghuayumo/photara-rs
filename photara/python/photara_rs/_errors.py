"""Exceptions raised by the public Photara API."""

from __future__ import annotations


class PhotaraError(RuntimeError):
    """Base class for renderer or session errors."""


class PhotaraClosedError(PhotaraError):
    """Raised when a Scene or Session is used after it was closed."""


class PhotaraRemovedError(PhotaraError):
    """Raised when a removed scene item is used again."""
