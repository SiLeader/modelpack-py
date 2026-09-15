"""Exceptions raised by the ModelPack client."""


class ModelPackError(Exception):
    """Base exception for this package."""


class InvalidModelPackError(ModelPackError):
    """Raised when an artifact does not conform to the ModelPack specification."""


class UnsafePathError(ModelPackError):
    """Raised when an artifact tries to write outside the destination directory."""
