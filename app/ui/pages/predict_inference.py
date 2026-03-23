from __future__ import annotations

import streamlit as st

from app.services.trainer import predict_with_model


def render() -> None:
    st.header("Predict / Inference")
    model = st.session_state.get("trained_model")
    windowed = st.session_state.get("windowed")

    if model is None or windowed is None:
        st.warning("Bitte zuerst Dataset bauen und Modell trainieren.")
        return

    mode = st.selectbox("Prediction Mode", ["single scenario", "single window", "batch"])

    if mode == "single scenario":
        sid = st.selectbox("scenario_id", sorted(windowed["scenario_id"].astype(str).unique()))
        feat = windowed[windowed["scenario_id"].astype(str) == sid].head(1)
    elif mode == "single window":
        idx = st.number_input("window row index", 0, max(0, len(windowed) - 1), 0, key="predict_window_row_index")
        feat = windowed.iloc[[int(idx)]]
    else:
        size = st.number_input("batch size", 1, len(windowed), min(64, len(windowed)), key="predict_batch_size")
        feat = windowed.head(int(size))

    if st.button("Run Inference", type="primary"):
        pred = predict_with_model(model, feat)
        out = feat.reset_index(drop=True).copy()
        out = out.join(pred)
        st.session_state["predictions"] = out
        st.dataframe(out, use_container_width=True)
