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

    checks = {
        "Pflichtspalten": not any("missing required columns" in x for x in report.errors),
        "Datentypprüfung": not any("not numeric" in x for x in report.errors),
        "scenario_id Konsistenz": not any("unknown scenario_id" in x for x in report.errors),
        "tunnel_id Konsistenz": not any("unknown tunnel_id" in x for x in report.errors),
        "timestamp Konsistenz": not any("timestamp mismatch" in x for x in report.errors),
        "Duplikate": not any("duplicate" in x for x in report.errors),
    }

    st.subheader("Passed Checks")
    for label, ok in checks.items():
        (st.success if ok else st.info)(f"{label}: {'PASS' if ok else 'FAIL'}")

    st.json(report.to_dict())
