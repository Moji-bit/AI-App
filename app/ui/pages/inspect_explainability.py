from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.services.explainability import attention_proxy, error_breakdown, feature_importance_df


def render() -> None:
    st.header("Inspect / Explainability")
    model = st.session_state.get("trained_model")
    windowed = st.session_state.get("windowed")
    preds = st.session_state.get("predictions")

    if model is None:
        st.warning("Bitte zuerst ein Modell trainieren.")
        return

    fi = feature_importance_df(model)
    st.subheader("Feature Importance")
    st.dataframe(fi, use_container_width=True)
    st.plotly_chart(px.bar(fi.head(20), x="feature", y="importance", title="Feature Importance"), use_container_width=True)

    if windowed is not None and not windowed.empty:
        st.subheader("Attention Visualization (Proxy)")
        sample = windowed.select_dtypes(include="number").head(50)
        att = attention_proxy(sample)
        st.dataframe(att.head(30), use_container_width=True)

    if preds is not None and "target_event_type" in preds.columns:
        st.subheader("FP/FN Analyse")
        eb = error_breakdown(preds)
        st.dataframe(eb)
        st.plotly_chart(px.bar(eb, x="type", y="count", title="False Positive / False Negative"), use_container_width=True)
