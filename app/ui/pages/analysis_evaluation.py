from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.services.evaluator import compute_auc_metrics, compute_operational_metrics, confusion_matrix_df, evaluate_predictions


def render() -> None:
    st.header("Analysis / Evaluation")
    preds = st.session_state.get("predictions")
    if preds is None:
        st.warning("Bitte zuerst Predict / Inference ausführen.")
        return

    if "target_event_type" not in preds.columns:
        st.warning("Predictions enthalten keine Ground-Truth-Spalte 'target_event_type'.")
        return

    metrics = evaluate_predictions(preds["target_event_type"], preds["predicted_event_type"])
    auc = compute_auc_metrics(preds)
    ops = compute_operational_metrics(preds)

    st.subheader("Klassische Metriken")
    st.json(metrics)
    st.subheader("ROC/PR")
    st.json(auc)
    st.subheader("Operational Metrics")
    st.json(ops)

    cm = confusion_matrix_df(preds["target_event_type"], preds["predicted_event_type"])
    st.subheader("Confusion Matrix")
    st.dataframe(cm, use_container_width=True)
    st.plotly_chart(px.imshow(cm, text_auto=True, title="Confusion Matrix"), use_container_width=True)
