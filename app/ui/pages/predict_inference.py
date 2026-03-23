from __future__ import annotations

import streamlit as st

from ...services.trainer import predict_with_model


def render() -> None:
    st.header("Predict / Inference")
    model = st.session_state.get("trained_model")
    windowed = st.session_state.get("windowed")

    if model is None or windowed is None or windowed.empty:
        st.warning("Bitte zuerst Dataset bauen und Modell trainieren.")
        return

    mode = st.selectbox("Prediction Mode", ["single scenario", "single window", "batch"])

    if mode == "single scenario":
        ids = sorted(windowed["scenario_id"].astype(str).unique())
        sid = st.selectbox("scenario_id", ids)
        feat = windowed[windowed["scenario_id"].astype(str) == sid].head(1)
    elif mode == "single window":
        idx = st.number_input("window row index", 0, max(0, len(windowed) - 1), 0)
        feat = windowed.iloc[[int(idx)]]
    else:
        size = st.number_input("batch size", 1, len(windowed), min(64, len(windowed)))
        feat = windowed.head(int(size))

    if st.button("Run Inference", type="primary"):
        try:
            pred = predict_with_model(model, feat)
        except Exception as exc:
            st.error(f"Inference fehlgeschlagen: {exc}")
            return
        out = feat.reset_index(drop=True).copy()
        out = out.join(pred)
        st.session_state["predictions"] = out
        st.dataframe(out, use_container_width=True)
