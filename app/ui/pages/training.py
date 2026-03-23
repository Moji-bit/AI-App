from __future__ import annotations

import json

import plotly.express as px
import streamlit as st

from app.services.trainer import TrainingConfig, train_model


def render() -> None:
    st.header("Training")

    splits = st.session_state.get("splits")
    model_config = st.session_state.get("model_config")
    windowed = st.session_state.get("windowed")

    if windowed is None:
        st.warning("Bitte zuerst Training Dataset Builder ausführen.")
        return
    if model_config is None:
        st.warning("Bitte zuerst Model Architecture konfigurieren.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.number_input("epochs", 1, 500, 40, key="training_epochs")
        batch_size = st.number_input("batch_size", 1, 4096, 128, key="training_batch_size")
        learning_rate = st.number_input("learning_rate", 0.00001, 1.0, 0.001, format="%.5f", key="training_learning_rate")
    with c2:
        optimizer = st.selectbox("optimizer", ["adam", "adamw", "sgd"])
        scheduler = st.selectbox("scheduler", ["none", "cosine", "step"])
        early_stopping = st.checkbox("early stopping", value=True)
    with c3:
        patience = st.number_input("patience", 1, 100, 8, key="training_patience")
        use_cuda = st.checkbox("use_cuda", value=False)
        random_seed = st.number_input("random_seed", 0, 10_000_000, 42, key="training_random_seed")

    class_weights = st.text_area("class weights (JSON)", value='{"normal":1.0,"accident":2.0,"fire":3.0}')
    checkpoint_saving = st.checkbox("checkpoint saving", value=True)

    if st.button("Train Model", type="primary"):
        cfg = TrainingConfig(
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=bool(early_stopping),
            patience=int(patience),
            use_cuda=bool(use_cuda),
            random_seed=int(random_seed),
            class_weights={str(k): float(v) for k, v in json.loads(class_weights).items()},
            checkpoint_saving=bool(checkpoint_saving),
        )

        label_mode = "target_event" if "target_event" in windowed.columns else "target"
        model, history, summary = train_model(
            splits["train"] if splits else windowed,
            splits["val"] if splits else windowed,
            model_config,
            cfg,
            "multi_task" if "target_event" in windowed.columns else label_mode,
        )

        st.session_state["trained_model"] = model
        st.session_state["training_history"] = history
        st.session_state["training_summary"] = summary
        st.success("Training abgeschlossen.")

    history = st.session_state.get("training_history")
    summary = st.session_state.get("training_summary")

    if history is not None:
        fig = px.line(history, x="epoch", y=["train_loss", "val_loss"], title="Training/Validation Loss")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(history, use_container_width=True)
    if summary:
        st.json(summary)
