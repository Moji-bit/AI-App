from __future__ import annotations

import pandas as pd

from app.services.model_factory import ModelConfig
from app.services.trainer import TrainingConfig, train_model


def test_train_model_without_target_tte_column() -> None:
    train_df = pd.DataFrame(
        {
            "scenario_id": ["A", "B", "C", "D"],
            "speed_mean_kmh": [70.0, 60.0, 55.0, 65.0],
            "occupancy_pct": [20.0, 35.0, 42.0, 30.0],
            "target": ["normal", "accident", "accident", "normal"],
        }
    )
    val_df = train_df.copy()

    model, history, summary = train_model(
        train_df=train_df,
        val_df=val_df,
        model_config=ModelConfig(model_type="LSTM"),
        training_config=TrainingConfig(epochs=5, patience=2),
        label_mode="event_classification",
    )

    assert not history.empty
    assert summary["epochs_completed"] >= 1
    assert model.feature_importance
