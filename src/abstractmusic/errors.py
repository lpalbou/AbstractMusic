"""
AbstractMusic errors.

This module is intentionally dependency-light.
"""

from __future__ import annotations


class AbstractMusicError(RuntimeError):
    """Base class for AbstractMusic errors."""


class OptionalDependencyMissingError(AbstractMusicError):
    """Raised when an optional dependency is missing or failed to import."""


class BackendNotConfiguredError(AbstractMusicError):
    """Raised when no backend is configured on a manager/capability."""


class CapabilityNotSupportedError(AbstractMusicError):
    """Raised when a backend does not support a requested operation/parameter."""

