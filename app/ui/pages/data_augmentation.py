from __future__ import annotations

import json

import streamlit as st

from app.services.augmentation_engine import AugmentationConfig, PRESETS, generate_augmented_dataset


def render() -> None:
    st.header("Data Augmentation")
    frames = st.session_state.get("frames")
    report = st.session_state.get("validation_report")

    if not frames:
        st.warning("Bitte zuerst Daten importieren.")
        return
    if report and not report.schema_ok:
        st.error("Bitte zuerst Validierungsfehler beheben.")
        return

    preset = st.selectbox("Preset", ["custom", *PRESETS.keys()])
    cfg = AugmentationConfig()

    c1, c2, c3 = st.columns(3)
    with c1:
        target_scenarios = st.number_input("target scenario count", 1, 100000, cfg.target_scenarios, 100, key="data_aug_target_scenarios")
        augmentation_strength = st.slider("augmentation strength", 0.0, 1.0, cfg.augmentation_strength, 0.01)
        noise_level = st.slider("noise level", 0.0, 0.4, cfg.noise_level, 0.005)
    with c2:
        event_shift_range_s = st.number_input("event shift range", 0, 300, cfg.event_shift_range_s, 1, key="data_aug_event_shift_range")
        missing_rate = st.slider("missing rate", 0.0, 0.2, cfg.missing_rate, 0.001)
        outlier_rate = st.slider("outlier rate", 0.0, 0.05, cfg.outlier_rate, 0.001)
    with c3:
        seed = st.number_input("random seed", 0, 10_000_000, cfg.seed, 1, key="data_aug_seed")
        allowed_weather = st.multiselect("allowed weather variation", ["clear", "rain", "snow", "fog", "storm"], cfg.allowed_weather)

    class_balance_text = st.text_area("class balance target (JSON)", json.dumps(cfg.class_balance_targets, indent=2), height=160)

    if st.button("Run Augmentation", type="primary"):
        class_targets = {str(k): float(v) for k, v in json.loads(class_balance_text).items()}
        run_cfg = AugmentationConfig(
            target_scenarios=int(target_scenarios),
            augmentation_strength=float(augmentation_strength),
            noise_level=float(noise_level),
            event_shift_range_s=int(event_shift_range_s),
            missing_rate=float(missing_rate),
            outlier_rate=float(outlier_rate),
            allowed_weather=list(allowed_weather),
            class_balance_targets=class_targets,
            seed=int(seed),
        )
        augmented = generate_augmented_dataset(
            frames["scenario_metadata"],
            frames["timeseries"],
            frames["ground_truth"],
            run_cfg,
        )
        st.session_state["augmented"] = augmented
        st.success("Augmentation abgeschlossen.")

    augmented = st.session_state.get("augmented")
    if augmented:
        st.write("Augmented scenario count:", len(augmented["augmented_scenario_metadata"]))
        st.dataframe(augmented["augmented_scenario_metadata"].head(50), use_container_width=True)
