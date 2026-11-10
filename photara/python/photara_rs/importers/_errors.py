"""Errors raised by the asset importers."""

from __future__ import annotations


class PhotaraImportError(Exception):
    """Base class for asset import failures."""


class ObjImportError(PhotaraImportError):
    """Raised for malformed or unsupported OBJ/MTL input."""

    def __init__(self, message: str, *, path: str | None = None, line: int | None = None):
        location = ""
        if path:
            location = str(path)
        if line is not None:
            location = f"{location}:{line}" if location else f"line {line}"
        super().__init__(f"{location}: {message}" if location else message)
        self.path = path
        self.line = line


class GltfImportError(PhotaraImportError):
    """Raised for malformed or unsupported glTF input."""

    def __init__(self, message: str, *, path: str | None = None, line: int | None = None):
        location = ""
        if path:
            location = str(path)
        if line is not None:
            location = f"{location}:{line}" if location else f"line {line}"
        super().__init__(f"{location}: {message}" if location else message)
        self.path = path
        self.line = line
