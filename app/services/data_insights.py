from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PresetRecommendation:
    preset: str
    reason: str
    confidence: float


def recommend_augmentation_preset(scenario_metadata: pd.DataFrame, ground_truth: pd.DataFrame | None = None) -> PresetRecommendation:
    if scenario_metadata is None or scenario_metadata.empty:
        return PresetRecommendation("custom", "Keine Szenario-Metadaten verfügbar", 0.0)

    event_series = scenario_metadata.get("event_type", pd.Series(dtype=str)).astype(str).str.lower()
    if ground_truth is not None and not ground_truth.empty and "label_event_type" in ground_truth.columns:
        gt_event = ground_truth["label_event_type"].astype(str).str.lower()
        event_series = pd.concat([event_series, gt_event[gt_event != "normal"]], ignore_index=True)

    if event_series.empty:
        return PresetRecommendation("normal traffic", "Keine Event-Hinweise gefunden", 0.4)

    freq = event_series.value_counts(normalize=True)
    top_event = str(freq.index[0])
    top_share = float(freq.iloc[0])

    if len(freq) > 1 and float(freq.iloc[:2].sum()) > 0.75 and top_share < 0.65:
        return PresetRecommendation(
            "mixed disturbance",
            f"Mehrere dominante Events erkannt ({', '.join(freq.index[:3])})",
            min(0.95, float(freq.iloc[:2].sum())),
        )

    fire_aliases = {"fire", "vehicle_fire"}
    if top_event in fire_aliases:
        return PresetRecommendation("fire", f"Brand-Ereignisse dominieren ({top_share:.1%})", top_share)

    mapping = {
        "normal": "normal traffic",
        "congestion": "congestion",
        "accident": "accident",
        "sensor_fault": "sensor fault",
    }
    return PresetRecommendation(mapping.get(top_event, "custom"), f"Dominantes Event: {top_event} ({top_share:.1%})", top_share)


def detect_leakage_risk(windowed: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> list[str]:
    findings: list[str] = []
    split_scenarios = {k: set(v.get("scenario_id", pd.Series(dtype=str)).astype(str).unique()) for k, v in splits.items()}

    inter_train_val = split_scenarios.get("train", set()) & split_scenarios.get("val", set())
    inter_train_test = split_scenarios.get("train", set()) & split_scenarios.get("test", set())
    inter_val_test = split_scenarios.get("val", set()) & split_scenarios.get("test", set())
    if inter_train_val or inter_train_test or inter_val_test:
        findings.append("Szenario-IDs überlappen zwischen Splits (Leakage-Risiko)")

    if {"scenario_id", "window_start_s", "window_end_s"}.issubset(windowed.columns):
        duplicate_windows = int(
            windowed[["scenario_id", "window_start_s", "window_end_s"]].duplicated().sum()
        )
        if duplicate_windows > 0:
            findings.append(f"{duplicate_windows} doppelte Window-Segmente erkannt")

    return findings


def split_summary(splits: dict[str, pd.DataFrame], label_col: str = "target_event_type") -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name, df in splits.items():
        label_distribution = (
            df[label_col].astype(str).value_counts(normalize=True).round(4).to_dict() if label_col in df.columns else {}
        )
        summary[split_name] = {
            "samples": int(len(df)),
            "scenarios": int(df["scenario_id"].astype(str).nunique()) if "scenario_id" in df.columns else 0,
            "class_distribution": label_distribution,
        }
    return summary


def profile_dataset(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    ts = frames.get("timeseries", pd.DataFrame())
    gt = frames.get("ground_truth", pd.DataFrame())

    num_cols = ts.select_dtypes(include="number").columns.tolist()
    correlation = ts[num_cols].corr(numeric_only=True).replace([np.inf, -np.inf], np.nan) if num_cols else pd.DataFrame()
    missing_ratio = (ts.isna().mean().sort_values(ascending=False).head(15).round(4).to_dict() if not ts.empty else {})

    event_distribution = (
        gt["label_event_type"].astype(str).value_counts(normalize=True).round(4).to_dict()
        if "label_event_type" in gt.columns
        else {}
    )

    outliers: dict[str, int] = {}
    for col in num_cols[:20]:
        q1, q3 = ts[col].quantile(0.25), ts[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers[col] = int(((ts[col] < lo) | (ts[col] > hi)).sum())

    return {
        "event_distribution": event_distribution,
        "missing_ratio": missing_ratio,
        "correlation": correlation,
        "outliers": outliers,
        "numeric_columns": num_cols,
    }
