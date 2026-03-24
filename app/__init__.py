"""Tunnel AI package.

The Shiny app object is exposed lazily via ``app`` so imports like
``app.services`` do not instantiate the web UI during test collection.
"""

from __future__ import annotations

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from app.shiny_app.application import app as shiny_app

        return shiny_app
    raise AttributeError(name)
