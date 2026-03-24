from __future__ import annotations

from shiny import App

from .server.handlers import register_handlers
from .ui.layout import build_layout

app_ui = build_layout()


def server(input, output, session):
    register_handlers(input, output, session)


app = App(app_ui, server)
