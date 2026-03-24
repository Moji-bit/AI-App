from __future__ import annotations

from pathlib import Path

# Allow this file-based entrypoint to coexist with the `app/` package.
__path__ = [str(Path(__file__).with_name("app"))]

from app.shiny_app.application import app  # type: ignore  # noqa: E402

__all__ = ["app"]
