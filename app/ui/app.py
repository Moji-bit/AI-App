from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import streamlit as st


def _bootstrap_package_imports() -> None:
    """Ensure `app` and `app.ui` are importable when Streamlit executes this file directly."""
    repo_root = Path(__file__).resolve().parents[2]
    app_dir = repo_root / "app"
    ui_dir = app_dir / "ui"

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if "app" not in sys.modules:
        app_pkg = types.ModuleType("app")
        app_pkg.__path__ = [str(app_dir)]
        sys.modules["app"] = app_pkg

    if "app.ui" not in sys.modules:
        ui_pkg = types.ModuleType("app.ui")
        ui_pkg.__path__ = [str(ui_dir)]
        sys.modules["app.ui"] = ui_pkg


def _load_pages():
    if __package__ not in {None, ""}:
        from .pages import load_pages

        return load_pages(__package__ + ".pages")

    _bootstrap_package_imports()
    pages_module = importlib.import_module("app.ui.pages")
    return pages_module.load_pages("app.ui.pages")


st.set_page_config(page_title="Tunnel Time-Series AI App", layout="wide")
st.title("Tunnel Time-Series AI App")

loaded_pages = _load_pages()

tabs = st.tabs([name for name, _ in loaded_pages])
for tab, (_, module) in zip(tabs, loaded_pages):
    with tab:
        module.render()
