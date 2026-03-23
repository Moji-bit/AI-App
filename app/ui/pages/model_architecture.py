from __future__ import annotations

import streamlit as st

from ...services.model_factory import MODEL_PRESETS, ModelConfig, available_model_types, config_from_preset


def render() -> None:
    st.header("Model Architecture")

    preset = st.selectbox("Preset", list(MODEL_PRESETS.keys()))
    cfg = config_from_preset(preset)

    c1, c2, c3 = st.columns(3)
    with c1:
        model_type = st.selectbox("model", available_model_types(), index=max(0, available_model_types().index(cfg.model_type) if cfg.model_type in available_model_types() else 0))
        input_dim = st.number_input("input_dim", 1, 4096, cfg.input_dim)
        hidden_dim = st.number_input("hidden_dim", 4, 2048, cfg.hidden_dim)
        d_model = st.number_input("d_model", 8, 2048, cfg.d_model)
    with c2:
        num_layers = st.number_input("num_layers", 1, 24, cfg.num_layers)
        num_heads = st.number_input("num_heads", 1, 16, cfg.num_heads)
        dropout = st.slider("dropout", 0.0, 0.8, cfg.dropout, 0.01)
    with c3:
        sequence_length = st.number_input("sequence_length", 3, 300, cfg.sequence_length)
        forecast_horizon = st.number_input("forecast_horizon", 1, 300, cfg.forecast_horizon)
        output_heads = st.multiselect("output heads", ["event", "risk", "tte"], default=cfg.output_heads or ["event"])

    loss_json = st.text_area("loss weights (JSON)", '{"event":1.0,"risk":0.5,"tte":0.2}')

    if st.button("Save Architecture", type="primary"):
        import json

        model_cfg = ModelConfig(
            model_type=model_type,
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            d_model=int(d_model),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
            sequence_length=int(sequence_length),
            forecast_horizon=int(forecast_horizon),
            output_heads=list(output_heads),
            loss_weights={str(k): float(v) for k, v in json.loads(loss_json).items()},
        )
        st.session_state["model_config"] = model_cfg
        st.success("Model config gespeichert.")

    current = st.session_state.get("model_config")
    if current:
        st.download_button("Export JSON", data=current.to_json(), file_name="model_config.json")
        imported = st.text_area("Import JSON", value=current.to_json(), height=220)
        if st.button("Load JSON"):
            st.session_state["model_config"] = ModelConfig.from_json(imported)
            st.success("JSON geladen.")
