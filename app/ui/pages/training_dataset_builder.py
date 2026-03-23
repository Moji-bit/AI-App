from __future__ import annotations

import streamlit as st

from app.services.data_merger import build_merged_dataset
from app.services.dataset_builder import DatasetBuildConfig, add_training_targets, build_windowed_training_dataset, train_val_test_split
from app.services.export_service import export_augmented_files


def render() -> None:
    st.header("Training Dataset Builder")
    frames = st.session_state.get("frames")
    augmented = st.session_state.get("augmented")
    if not frames:
        st.warning("Bitte zuerst Daten importieren.")
        return

    use_augmented = st.checkbox("Augmented Daten verwenden", value=bool(augmented))
    src = frames.copy()
    if use_augmented and augmented:
        src = {
            "tunnel_config": frames["tunnel_config"],
            "scenario_metadata": augmented["augmented_scenario_metadata"],
            "timeseries": augmented["augmented_timeseries"],
            "ground_truth": augmented["augmented_ground_truth"],
        }

    sequence_length = st.number_input("sequence_length", 3, 300, 30)
    forecast_horizon = st.number_input("forecast_horizon", 1, 300, 5)
    stride = st.number_input("stride", 1, 100, 5)
    label_mode = st.selectbox("Label-Modus", ["event_classification", "risk_classification", "time_to_event_regression", "multi_task"])

    split_cols = st.columns(3)
    train_ratio = split_cols[0].slider("train", 0.1, 0.9, 0.7, 0.05)
    val_ratio = split_cols[1].slider("val", 0.05, 0.4, 0.15, 0.05)
    test_ratio = 1.0 - train_ratio - val_ratio
    split_cols[2].write(f"test: {test_ratio:.2f}")

    if st.button("Build Training Dataset", type="primary"):
        merged = build_merged_dataset(src)
        cfg = DatasetBuildConfig(
            sequence_length=int(sequence_length),
            forecast_horizon=int(forecast_horizon),
            stride=int(stride),
            label_mode=label_mode,
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
        )
        windowed = build_windowed_training_dataset(merged, cfg)
        windowed = add_training_targets(windowed, label_mode)
        splits = train_val_test_split(windowed, cfg)

        st.session_state["merged"] = merged
        st.session_state["windowed"] = windowed
        st.session_state["splits"] = splits
        st.success("Training Dataset erstellt.")

    windowed = st.session_state.get("windowed")
    splits = st.session_state.get("splits")
    if windowed is not None:
        st.write(f"Windowed samples: {len(windowed)}")
        st.dataframe(windowed.head(100), use_container_width=True)

    if splits:
        st.write({k: len(v) for k, v in splits.items()})

    out_dir = st.text_input("Export-Verzeichnis", "exports")
    if st.button("Export Training Files") and st.session_state.get("merged") is not None:
        if augmented:
            files = export_augmented_files(
                out_dir,
                augmented["augmented_scenario_metadata"],
                augmented["augmented_timeseries"],
                augmented["augmented_ground_truth"],
                augmented_tunnel_config=frames["tunnel_config"],
                merged_training_dataset=st.session_state["merged"],
            )
            st.json({k: str(v) for k, v in files.items()})
        if windowed is not None:
            window_path = f"{out_dir}/windowed_dataset.csv"
            windowed.to_csv(window_path, index=False)
            st.info(f"Windowed export: {window_path}")
