"""Tunnel AI package.

Expose a concrete module-level ``app`` attribute so ASGI loaders that use
static attribute lookup (instead of ``__getattr__``) can resolve `app:app`.
"""

from app.shiny_app.application import app

__all__ = ["app"]
