from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


LabelMode = Literal["event_classification", "risk_classification", "time_to_event_regression", "multi_task"]


@dataclass
class DatasetBuildConfig:
    sequence_length: int = 30
    forecast_horizon: int = 5
    stride: int = 5
    label_mode: LabelMode = "event_classification"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


NUMERIC_FEATURES = [
    "speed_mean_kmh",
    "flow_veh_h",
    "occupancy_pct",
    "queue_length_m",
    "stopped_vehicle_count",
    "co_ppm",
    "no2_ppm",
    "pm25_ug_m3",
    "visibility_m",
    "jet_fan_power_pct",
]

BASE_WINDOWED_COLUMNS = [
    "scenario_id",
    "window_start_s",
    "window_end_s",
    "sequence_length",
    "forecast_horizon",
    "target_event_type",
    "target_risk_level",
    "target_time_to_event_s",
]


def _time_to_event(group: pd.DataFrame) -> pd.Series:
    active = pd.to_numeric(group.get("label_event_active", 0), errors="coerce").fillna(0).astype(int)
    idx_active = np.where(active.to_numpy() == 1)[0]
    t = pd.to_numeric(group["timestamp_s"], errors="coerce").to_numpy()
    if len(idx_active) == 0:
        return pd.Series(np.full(len(group), np.nan), index=group.index)
    first = idx_active[0]
    return pd.Series(np.maximum(t[first] - t, 0), index=group.index)


def build_windowed_training_dataset(merged: pd.DataFrame, config: DatasetBuildConfig) -> pd.DataFrame:
    rows: list[dict] = []

    for scenario_id, group in merged.groupby("scenario_id"):
        g = group.sort_values("timestamp_s").reset_index(drop=True).copy()
        g["time_to_event_s"] = _time_to_event(g)

        for start in range(0, max(0, len(g) - config.sequence_length - config.forecast_horizon + 1), config.stride):
            seq = g.iloc[start : start + config.sequence_length]
            target_ix = start + config.sequence_length - 1 + config.forecast_horizon
            target = g.iloc[min(target_ix, len(g) - 1)]

            features = {f"{c}_mean": float(pd.to_numeric(seq[c], errors="coerce").mean()) for c in NUMERIC_FEATURES if c in seq.columns}
            features.update({f"{c}_std": float(pd.to_numeric(seq[c], errors="coerce").std()) for c in NUMERIC_FEATURES if c in seq.columns})

            row = {
                "scenario_id": scenario_id,
                "window_start_s": float(seq["timestamp_s"].min()),
                "window_end_s": float(seq["timestamp_s"].max()),
                "sequence_length": config.sequence_length,
                "forecast_horizon": config.forecast_horizon,
                "target_event_type": str(target.get("label_event_type", "normal")),
                "target_risk_level": str(target.get("label_risk_level", "low")),
                "target_time_to_event_s": float(target.get("time_to_event_s", np.nan)),
            }
            row.update(features)
            rows.append(row)

    # Keep engineered feature columns instead of truncating to base metadata columns.
    windowed = pd.DataFrame(rows)
    if windowed.empty:
        return pd.DataFrame(columns=BASE_WINDOWED_COLUMNS)

    ordered = BASE_WINDOWED_COLUMNS + [c for c in windowed.columns if c not in BASE_WINDOWED_COLUMNS]
    return windowed[ordered]


def add_training_targets(windowed: pd.DataFrame, mode: LabelMode) -> pd.DataFrame:
    out = windowed.copy()
    required = {
        "event_classification": ["target_event_type"],
        "risk_classification": ["target_risk_level"],
        "time_to_event_regression": ["target_time_to_event_s"],
        "multi_task": ["target_event_type", "target_risk_level", "target_time_to_event_s"],
    }[mode]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"Windowed dataset is missing required columns for '{mode}': {', '.join(missing)}"
        )

    if mode == "event_classification":
        out["target"] = out["target_event_type"]
    elif mode == "risk_classification":
        out["target"] = out["target_risk_level"]
    elif mode == "time_to_event_regression":
        out["target"] = pd.to_numeric(out["target_time_to_event_s"], errors="coerce")
    else:
        out["target_event"] = out["target_event_type"]
        out["target_risk"] = out["target_risk_level"]
        out["target_tte"] = pd.to_numeric(out["target_time_to_event_s"], errors="coerce")
    return out


def train_val_test_split(windowed: pd.DataFrame, config: DatasetBuildConfig) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config.random_seed)
    scenario_ids = windowed["scenario_id"].dropna().astype(str).unique().tolist()
    shuffled = np.array(scenario_ids, dtype=object)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])
    test_ids = set(shuffled[n_train + n_val :])

    return {
        "train": windowed[windowed["scenario_id"].astype(str).isin(train_ids)].reset_index(drop=True),
        "val": windowed[windowed["scenario_id"].astype(str).isin(val_ids)].reset_index(drop=True),
        "test": windowed[windowed["scenario_id"].astype(str).isin(test_ids)].reset_index(drop=True),
    }
