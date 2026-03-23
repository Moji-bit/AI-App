from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.services.augmentation_engine import AugmentationConfig, generate_augmented_dataset
from app.services.data_loader import load_ground_truth, load_scenario_metadata, load_timeseries, load_tunnel_config
from app.services.data_merger import build_merged_dataset
from app.services.export_service import export_augmented_files
from app.services.scenario_generator import build_scenario_summary, create_windowed_sequences, preset_to_config
from app.services.schema_validator import validate_schema


st.set_page_config(page_title="Tunnel Dataset Builder", layout="wide")
st.title("Tunnel Dataset Builder + Data Augmentation")

st.markdown("Lade 4 CSV-Dateien hoch und generiere tausende plausible Trainingsszenarien.")

col_up1, col_up2 = st.columns(2)
with col_up1:
    tunnel_file = st.file_uploader("tunnel_config.csv", type=["csv"])
    metadata_file = st.file_uploader("scenario_metadata.csv", type=["csv"])
with col_up2:
    timeseries_file = st.file_uploader("timeseries.csv", type=["csv"])
    ground_truth_file = st.file_uploader("ground_truth.csv", type=["csv"])

frames = {}
if all([tunnel_file, metadata_file, timeseries_file, ground_truth_file]):
    frames["tunnel_config"] = load_tunnel_config(tunnel_file)
    frames["scenario_metadata"] = load_scenario_metadata(metadata_file)
    frames["timeseries"] = load_timeseries(timeseries_file)
    frames["ground_truth"] = load_ground_truth(ground_truth_file)

    st.subheader("Datenvorschau")
    tabs = st.tabs(["tunnel_config", "scenario_metadata", "timeseries", "ground_truth"])
    for t, name in zip(tabs, ["tunnel_config", "scenario_metadata", "timeseries", "ground_truth"]):
        with t:
            st.dataframe(frames[name].head(50), use_container_width=True)

    report = validate_schema(frames)
    st.subheader("Schema-Validierungsreport")
    st.json(report.to_dict())

    if report.schema_ok:
        st.success("Schema und Cross-File-Konsistenz sind gültig.")
        summary = build_scenario_summary(frames["scenario_metadata"])
        st.subheader("Szenario-Zusammenfassung")
        st.json(summary)

        st.subheader("Augmentation-Panel")
        preset = st.selectbox("Preset", list({"custom": {}, **{k: {} for k in [
            "normal traffic",
            "congestion",
            "accident",
            "fire",
            "sensor fault",
            "winter weather",
            "heavy rain",
            "mixed disturbance",
        ]}}.keys()))

        cfg = AugmentationConfig()
        if preset != "custom":
            cfg = preset_to_config(preset, cfg)

        c1, c2, c3 = st.columns(3)
        with c1:
            target_scenarios = st.number_input("Anzahl Ziel-Szenarien", min_value=1, value=cfg.target_scenarios, step=100)
            augmentation_strength = st.slider("Augmentationsstärke", 0.0, 1.0, float(cfg.augmentation_strength), 0.01)
            noise_level = st.slider("Noise-Level", 0.0, 0.5, float(cfg.noise_level), 0.005)
        with c2:
            event_shift_range_s = st.number_input("Event-Shift-Range (s)", min_value=0, value=cfg.event_shift_range_s, step=1)
            missing_rate = st.slider("Missing-Rate", 0.0, 0.2, float(cfg.missing_rate), 0.001)
            outlier_rate = st.slider("Outlier-Rate", 0.0, 0.05, float(cfg.outlier_rate), 0.001)
        with c3:
            seed = st.number_input("Random Seed", min_value=0, value=cfg.seed, step=1)
            allowed_weather = st.multiselect(
                "Erlaubte Wettervariationen",
                options=["clear", "rain", "snow", "fog", "storm"],
                default=cfg.allowed_weather,
            )

        default_targets = json.dumps(cfg.class_balance_targets, indent=2)
        class_balance_json = st.text_area("Class-Balance-Ziele (JSON)", value=default_targets, height=160)

        if st.button("Generate Augmented Dataset", type="primary"):
            parsed_targets = json.loads(class_balance_json)
            run_cfg = AugmentationConfig(
                target_scenarios=int(target_scenarios),
                augmentation_strength=float(augmentation_strength),
                noise_level=float(noise_level),
                event_shift_range_s=int(event_shift_range_s),
                missing_rate=float(missing_rate),
                outlier_rate=float(outlier_rate),
                allowed_weather=list(allowed_weather),
                class_balance_targets={str(k): float(v) for k, v in parsed_targets.items()},
                seed=int(seed),
            )

            augmented = generate_augmented_dataset(
                frames["scenario_metadata"],
                frames["timeseries"],
                frames["ground_truth"],
                run_cfg,
            )

            merged = build_merged_dataset(
                {
                    "scenario_metadata": augmented["augmented_scenario_metadata"],
                    "timeseries": augmented["augmented_timeseries"],
                    "ground_truth": augmented["augmented_ground_truth"],
                    "tunnel_config": frames["tunnel_config"],
                }
            )
            st.session_state["augmented"] = augmented
            st.session_state["merged"] = merged

        if "augmented" in st.session_state:
            augmented = st.session_state["augmented"]
            merged = st.session_state["merged"]

            st.subheader("Preview before export")
            st.write("Original vs. Augmentiert")

            orig_sid = st.selectbox("Original scenario_id", sorted(frames["scenario_metadata"]["scenario_id"].astype(str).unique()))
            aug_sid = st.selectbox("Augmented scenario_id", sorted(augmented["augmented_scenario_metadata"]["scenario_id"].astype(str).unique()))

            orig_ts = frames["timeseries"][frames["timeseries"]["scenario_id"].astype(str) == orig_sid]
            aug_ts = augmented["augmented_timeseries"][augmented["augmented_timeseries"]["scenario_id"].astype(str) == aug_sid]

            for metric in ["speed_mean_kmh", "flow_veh_h", "occupancy_pct", "co_ppm", "visibility_m", "queue_length_m"]:
                if metric in orig_ts.columns and metric in aug_ts.columns:
                    compare = pd.concat(
                        [
                            orig_ts[["timestamp_s", metric]].assign(source="original"),
                            aug_ts[["timestamp_s", metric]].assign(source="augmented"),
                        ],
                        ignore_index=True,
                    )
                    fig = px.line(compare, x="timestamp_s", y=metric, color="source", title=metric)
                    st.plotly_chart(fig, use_container_width=True)

            st.dataframe(augmented["augmented_scenario_metadata"].head(50), use_container_width=True)

            build_windows = st.checkbox("Windowed Sequences für Training erzeugen", value=False)
            if build_windows:
                window_size = st.number_input("Window Size", min_value=3, value=30, step=1)
                stride = st.number_input("Stride", min_value=1, value=5, step=1)
                windowed = create_windowed_sequences(merged, window_size=int(window_size), stride=int(stride))
                st.dataframe(windowed.head(100), use_container_width=True)
                st.session_state["windowed"] = windowed

            output_dir = st.text_input("Export-Verzeichnis", value=str(Path.cwd() / "exports"))
            if st.button("Exportiere Datensätze"):
                files = export_augmented_files(
                    output_dir=output_dir,
                    augmented_scenario_metadata=augmented["augmented_scenario_metadata"],
                    augmented_timeseries=augmented["augmented_timeseries"],
                    augmented_ground_truth=augmented["augmented_ground_truth"],
                    augmented_tunnel_config=frames["tunnel_config"],
                    merged_training_dataset=merged,
                )
                if "windowed" in st.session_state:
                    window_path = Path(output_dir) / "windowed_sequences.csv"
                    st.session_state["windowed"].to_csv(window_path, index=False)
                    st.info(f"Windowed dataset exportiert: {window_path}")
                st.success("Export abgeschlossen")
                st.json({k: str(v) for k, v in files.items()})
else:
    st.info("Bitte alle 4 CSV-Dateien hochladen, um zu starten.")
