from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from .augmentation_engine import AugmentationConfig, PRESETS


def preset_to_config(preset_name: str, base: AugmentationConfig | None = None) -> AugmentationConfig:
    config = base or AugmentationConfig()
    preset = PRESETS.get(preset_name.lower())
    if not preset:
        return config

    values = asdict(config)
    values.update({k: v for k, v in preset.items() if k in values})
    return AugmentationConfig(**values)


def build_scenario_summary(scenario_metadata: pd.DataFrame) -> dict[str, Any]:
    durations = pd.to_numeric(scenario_metadata.get("simulation_duration_s", pd.Series(dtype=float)), errors="coerce")
    return {
        "scenario_count": int(len(scenario_metadata)),
        "event_type_distribution": scenario_metadata.get("event_type", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "weather_distribution": scenario_metadata.get("weather_type", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "severity_distribution": scenario_metadata.get("event_severity", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "duration_s": {
            "min": float(durations.min()) if not durations.empty else 0.0,
            "max": float(durations.max()) if not durations.empty else 0.0,
            "mean": float(durations.mean()) if not durations.empty else 0.0,
        },
    }


def create_windowed_sequences(
    merged_dataset: pd.DataFrame,
    window_size: int = 30,
    stride: int = 5,
    label_column: str = "label_event_type",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_id, group in merged_dataset.groupby("scenario_id"):
        group = group.sort_values("timestamp_s").reset_index(drop=True)
        for start in range(0, max(0, len(group) - window_size + 1), stride):
            window = group.iloc[start : start + window_size]
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "window_start_ts": float(window["timestamp_s"].min()),
                    "window_end_ts": float(window["timestamp_s"].max()),
                    "window_size": window_size,
                    "label": str(window[label_column].iloc[-1]) if label_column in window else "unknown",
                    "risk_target": str(window.get("label_risk_level", pd.Series(["low"])).iloc[-1]),
                    "speed_mean": float(pd.to_numeric(window.get("speed_mean_kmh"), errors="coerce").mean()),
                    "flow_mean": float(pd.to_numeric(window.get("flow_veh_h"), errors="coerce").mean()),
                    "occupancy_mean": float(pd.to_numeric(window.get("occupancy_pct"), errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)
