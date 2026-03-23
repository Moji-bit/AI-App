from __future__ import annotations

import streamlit as st

from app.services.data_loader import load_ground_truth, load_scenario_metadata, load_timeseries, load_tunnel_config


def render() -> None:
    st.header("Data Import")

    c1, c2 = st.columns(2)
    with c1:
        tunnel_file = st.file_uploader("tunnel_config.csv", type=["csv"], key="upload_tunnel")
        meta_file = st.file_uploader("scenario_metadata.csv", type=["csv"], key="upload_meta")
    with c2:
        ts_file = st.file_uploader("timeseries.csv", type=["csv"], key="upload_ts")
        gt_file = st.file_uploader("ground_truth.csv", type=["csv"], key="upload_gt")

    if all([tunnel_file, meta_file, ts_file, gt_file]):
        st.session_state["frames"] = {
            "tunnel_config": load_tunnel_config(tunnel_file),
            "scenario_metadata": load_scenario_metadata(meta_file),
            "timeseries": load_timeseries(ts_file),
            "ground_truth": load_ground_truth(gt_file),
        }
        st.success("Alle Dateien geladen.")

    frames = st.session_state.get("frames")
    if frames:
        tabs = st.tabs(list(frames.keys()))
        for tab, key in zip(tabs, frames.keys()):
            with tab:
                df = frames[key]
                st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
                st.write("Spalten:", list(df.columns))
                st.dataframe(df.head(100), use_container_width=True)
    else:
        st.info("Bitte alle vier CSV-Dateien hochladen.")
