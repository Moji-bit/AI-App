from __future__ import annotations

import pandas as pd

from app.services.data_merger import build_merged_dataset
from app.services.dataset_builder import DatasetBuildConfig, add_training_targets, build_windowed_training_dataset, train_val_test_split
from app.services.export_service import export_augmented_files
from app.services.model_factory import ModelConfig
from app.services.trainer import TrainingConfig, predict_with_model, train_model
from tests.test_dataset_builder import sample_frames


def test_windowed_dataset_contains_engineered_features() -> None:
    frames = sample_frames()
    merged = build_merged_dataset(frames)
    cfg = DatasetBuildConfig(sequence_length=3, forecast_horizon=1, stride=1)

    windowed = build_windowed_training_dataset(merged, cfg)

    engineered = [c for c in windowed.columns if c.endswith("_mean") or c.endswith("_std")]
    assert engineered, "Expected engineered feature columns in windowed dataset"


def test_end_to_end_pipeline_train_predict_export(tmp_path) -> None:
    frames = sample_frames()
    merged = build_merged_dataset(frames)
    cfg = DatasetBuildConfig(sequence_length=3, forecast_horizon=1, stride=1)
    windowed = add_training_targets(build_windowed_training_dataset(merged, cfg), "event_classification")
    splits = train_val_test_split(windowed, cfg)

    # Guard against tiny synthetic datasets where val may be empty
    train_df = splits["train"] if not splits["train"].empty else windowed
    val_df = splits["val"] if not splits["val"].empty else windowed

    model, history, summary = train_model(
        train_df=train_df,
        val_df=val_df,
        model_config=ModelConfig(model_type="LSTM"),
        training_config=TrainingConfig(epochs=5, patience=2, device="cpu"),
        label_mode="event_classification",
    )

    preds = predict_with_model(model, windowed.head(3), top_n=2)
    assert not preds.empty
    assert "confidence" in preds.columns
    assert "top_classes" in preds.columns

    export_files = export_augmented_files(
        output_dir=tmp_path,
        augmented_scenario_metadata=frames["scenario_metadata"],
        augmented_timeseries=frames["timeseries"],
        augmented_ground_truth=frames["ground_truth"],
        augmented_tunnel_config=frames["tunnel_config"],
        merged_training_dataset=merged,
        windowed_dataset=windowed,
        training_history=history,
        training_summary=summary,
        model_config_json=ModelConfig(model_type="LSTM").to_json(),
    )
    assert export_files
    for path in export_files.values():
        assert path.exists()

    history_metrics = {"accuracy", "precision", "recall", "f1"}
    assert history_metrics.issubset(set(history.columns))

    assert isinstance(summary.get("best_f1"), float)
    assert isinstance(summary.get("best_accuracy"), float)

    # verify model/prediction class probabilities are stable and sum to 1
    prob_cols = [c for c in preds.columns if c.startswith("prob_")]
    if prob_cols:
        sums = preds[prob_cols].sum(axis=1)
        assert pd.Series((sums.round(6) == 1.0)).all()
