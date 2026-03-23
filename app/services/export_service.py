from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_augmented_files(
    output_dir: str | Path,
    augmented_scenario_metadata: pd.DataFrame,
    augmented_timeseries: pd.DataFrame,
    augmented_ground_truth: pd.DataFrame,
    augmented_tunnel_config: pd.DataFrame | None = None,
    merged_training_dataset: pd.DataFrame | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}

    files["augmented_scenario_metadata"] = output_path / "augmented_scenario_metadata.csv"
    augmented_scenario_metadata.to_csv(files["augmented_scenario_metadata"], index=False)

    files["augmented_timeseries"] = output_path / "augmented_timeseries.csv"
    augmented_timeseries.to_csv(files["augmented_timeseries"], index=False)

    files["augmented_ground_truth"] = output_path / "augmented_ground_truth.csv"
    augmented_ground_truth.to_csv(files["augmented_ground_truth"], index=False)

    if augmented_tunnel_config is not None:
        files["augmented_tunnel_config"] = output_path / "augmented_tunnel_config.csv"
        augmented_tunnel_config.to_csv(files["augmented_tunnel_config"], index=False)

    if merged_training_dataset is not None:
        files["merged_training_dataset"] = output_path / "merged_training_dataset.csv"
        merged_training_dataset.to_csv(files["merged_training_dataset"], index=False)

    return files
