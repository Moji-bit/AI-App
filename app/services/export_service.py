from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import pandas as pd


def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def export_augmented_files(
    output_dir: str | Path,
    augmented_scenario_metadata: pd.DataFrame,
    augmented_timeseries: pd.DataFrame,
    augmented_ground_truth: pd.DataFrame,
    augmented_tunnel_config: pd.DataFrame | None = None,
    merged_training_dataset: pd.DataFrame | None = None,
    windowed_dataset: pd.DataFrame | None = None,
    training_history: pd.DataFrame | None = None,
    training_summary: dict | None = None,
    model_config_json: str | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = _ts()

    files: dict[str, Path] = {}

    files["augmented_scenario_metadata"] = output_path / f"augmented_scenario_metadata_{stamp}.csv"
    augmented_scenario_metadata.to_csv(files["augmented_scenario_metadata"], index=False)

    files["augmented_timeseries"] = output_path / f"augmented_timeseries_{stamp}.csv"
    augmented_timeseries.to_csv(files["augmented_timeseries"], index=False)

    files["augmented_ground_truth"] = output_path / f"augmented_ground_truth_{stamp}.csv"
    augmented_ground_truth.to_csv(files["augmented_ground_truth"], index=False)

    if augmented_tunnel_config is not None:
        files["augmented_tunnel_config"] = output_path / f"augmented_tunnel_config_{stamp}.csv"
        augmented_tunnel_config.to_csv(files["augmented_tunnel_config"], index=False)

    if merged_training_dataset is not None:
        files["merged_training_dataset"] = output_path / f"merged_training_dataset_{stamp}.csv"
        merged_training_dataset.to_csv(files["merged_training_dataset"], index=False)

    if windowed_dataset is not None:
        files["windowed_dataset"] = output_path / f"windowed_dataset_{stamp}.csv"
        windowed_dataset.to_csv(files["windowed_dataset"], index=False)

    if training_history is not None:
        files["training_history"] = output_path / f"training_history_{stamp}.csv"
        training_history.to_csv(files["training_history"], index=False)

    if training_summary is not None:
        files["training_summary"] = output_path / f"training_summary_{stamp}.json"
        files["training_summary"].write_text(json.dumps(training_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if model_config_json is not None:
        files["model_config"] = output_path / f"model_config_{stamp}.json"
        files["model_config"].write_text(model_config_json, encoding="utf-8")

    return files
