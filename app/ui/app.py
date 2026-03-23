from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from app.ui.pages import load_pages
else:
    from .pages import load_pages


st.set_page_config(page_title="Tunnel Time-Series AI App", layout="wide")
st.title("Tunnel Time-Series AI App")

loaded_pages = load_pages("app.ui.pages" if __package__ in {None, ""} else __package__ + ".pages")

tabs = st.tabs([name for name, _ in loaded_pages])
for tab, (_, module) in zip(tabs, loaded_pages):
    with tab:
        module.render()
