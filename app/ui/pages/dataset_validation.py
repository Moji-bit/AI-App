from __future__ import annotations

import streamlit as st

from app.services.schema_validator import validate_schema


def render() -> None:
    st.header("Dataset Validation")
    frames = st.session_state.get("frames")
    if not frames:
        st.warning("Bitte zuerst Data Import durchführen.")
        return

    report = validate_schema(frames)
    st.session_state["validation_report"] = report

    st.subheader("Errors")
    if report.errors:
        for e in report.errors:
            st.error(e)
    else:
        st.success("Keine Errors")

    st.subheader("Warnings")
    if report.warnings:
        for w in report.warnings:
            st.warning(w)
    else:
        st.success("Keine Warnings")

    st.subheader("Passed Checks")
    passed = [
        "Pflichtspalten" if not any("missing required columns" in x for x in report.errors) else None,
        "Datentyp-Checks" if not any("not numeric" in x for x in report.errors) else None,
        "scenario_id/tunnel_id/timestamp-Konsistenz" if not any("unknown" in x or "mismatch" in x for x in report.errors) else None,
    ]
    for p in [x for x in passed if x]:
        st.success(p)

    st.json(report.to_dict())
