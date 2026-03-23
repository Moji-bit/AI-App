from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .model_factory import ModelConfig

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import log_loss, mean_squared_error
except Exception:  # pragma: no cover
    RandomForestClassifier = None
    RandomForestRegressor = None
    log_loss = None
    mean_squared_error = None


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    scheduler: str = "none"
    early_stopping: bool = True
    patience: int = 5
    use_cuda: bool = False
    random_seed: int = 42
    class_weights: dict[str, float] | None = None
    checkpoint_saving: bool = True


@dataclass
class TrainedModel:
    model_config: ModelConfig
    training_config: TrainingConfig
    label_mode: str
    feature_columns: list[str]
    task_models: dict[str, Any]
    class_order: dict[str, list[str]]


FEATURE_EXCLUDE = {
    "scenario_id",
    "window_start_s",
    "window_end_s",
    "target",
    "target_event",
    "target_risk",
    "target_tte",
    "target_event_type",
    "target_risk_level",
    "target_time_to_event_s",
}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in FEATURE_EXCLUDE or c.startswith("target_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _ensure_ml_backend() -> None:
    if RandomForestClassifier is None or RandomForestRegressor is None:
        raise RuntimeError("scikit-learn is required for training/inference. Please install dependencies from requirements/environment.")


def _make_classifier(model_type: str, seed: int, class_weights: dict[str, float] | None = None) -> Any:
    trees = 150 if "Transformer" in model_type else 120
    depth = 12 if "Hybrid" in model_type else 10
    return RandomForestClassifier(
        n_estimators=trees,
        max_depth=depth,
        random_state=seed,
        class_weight=class_weights,
        n_jobs=-1,
    )


def _make_regressor(model_type: str, seed: int) -> Any:
    trees = 200 if "Transformer" in model_type else 120
    depth = 12 if "Hybrid" in model_type else 10
    return RandomForestRegressor(n_estimators=trees, max_depth=depth, random_state=seed, n_jobs=-1)


def _simulate_epoch_metrics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    is_regression: bool,
) -> tuple[float, float]:
    if train_df.empty or val_df.empty:
        return 0.0, 0.0

    X_train = train_df[feature_cols].fillna(0.0)
    X_val = val_df[feature_cols].fillna(0.0)

    if is_regression:
        y_train = pd.to_numeric(train_df[target_col], errors="coerce").fillna(train_df[target_col].median())
        y_val = pd.to_numeric(val_df[target_col], errors="coerce").fillna(val_df[target_col].median())
        m = RandomForestRegressor(n_estimators=30, random_state=7, n_jobs=-1)
        m.fit(X_train, y_train)
        train_loss = mean_squared_error(y_train, m.predict(X_train)) if mean_squared_error else float(np.mean((y_train - m.predict(X_train)) ** 2))
        val_loss = mean_squared_error(y_val, m.predict(X_val)) if mean_squared_error else float(np.mean((y_val - m.predict(X_val)) ** 2))
        return float(train_loss), float(val_loss)

    y_train = train_df[target_col].astype(str)
    y_val = val_df[target_col].astype(str)
    m = RandomForestClassifier(n_estimators=30, random_state=7, n_jobs=-1)
    m.fit(X_train, y_train)
    train_prob = m.predict_proba(X_train)
    val_prob = m.predict_proba(X_val)
    classes = list(m.classes_)
    train_loss = log_loss(y_train, train_prob, labels=classes) if log_loss else float(np.mean(y_train != m.predict(X_train)))
    val_loss = log_loss(y_val, val_prob, labels=classes) if log_loss else float(np.mean(y_val != m.predict(X_val)))
    return float(train_loss), float(val_loss)


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    label_mode: str,
) -> tuple[TrainedModel, pd.DataFrame, dict[str, Any]]:
    _ensure_ml_backend()

    feature_cols = _feature_columns(train_df)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for training.")

    history_rows: list[dict[str, float]] = []
    best_val = np.inf
    best_epoch = 1
    stagnant = 0

    for epoch in range(1, training_config.epochs + 1):
        if label_mode == "time_to_event_regression":
            t_loss, v_loss = _simulate_epoch_metrics(train_df, val_df, feature_cols, "target", is_regression=True)
        else:
            target_col = "target_event" if label_mode == "multi_task" else "target"
            t_loss, v_loss = _simulate_epoch_metrics(train_df, val_df, feature_cols, target_col, is_regression=False)

        history_rows.append({"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss})
        if v_loss < best_val:
            best_val = v_loss
            best_epoch = epoch
            stagnant = 0
        else:
            stagnant += 1

        if training_config.early_stopping and stagnant >= training_config.patience:
            break

    class_weights = training_config.class_weights
    task_models: dict[str, Any] = {}
    class_order: dict[str, list[str]] = {}

    X_train = train_df[feature_cols].fillna(0.0)

    if label_mode in {"event_classification", "risk_classification"}:
        clf = _make_classifier(model_config.model_type, training_config.random_seed, class_weights)
        y = train_df["target"].astype(str)
        clf.fit(X_train, y)
        task_models["target"] = clf
        class_order["target"] = list(clf.classes_)
    elif label_mode == "time_to_event_regression":
        reg = _make_regressor(model_config.model_type, training_config.random_seed)
        y = pd.to_numeric(train_df["target"], errors="coerce").fillna(train_df["target"].median())
        reg.fit(X_train, y)
        task_models["target"] = reg
    else:
        event_clf = _make_classifier(model_config.model_type, training_config.random_seed, class_weights)
        risk_clf = _make_classifier(model_config.model_type, training_config.random_seed + 1, None)
        tte_reg = _make_regressor(model_config.model_type, training_config.random_seed + 2)

        event_clf.fit(X_train, train_df["target_event"].astype(str))
        risk_clf.fit(X_train, train_df["target_risk"].astype(str))
        y_tte = pd.to_numeric(train_df["target_tte"], errors="coerce")
        y_tte = y_tte.fillna(y_tte.median())
        tte_reg.fit(X_train, y_tte)

        task_models = {"event": event_clf, "risk": risk_clf, "tte": tte_reg}
        class_order = {"event": list(event_clf.classes_), "risk": list(risk_clf.classes_)}

    model = TrainedModel(
        model_config=model_config,
        training_config=training_config,
        label_mode=label_mode,
        feature_columns=feature_cols,
        task_models=task_models,
        class_order=class_order,
    )

    history = pd.DataFrame(history_rows)
    best = history.loc[history["val_loss"].idxmin()].to_dict() if not history.empty else {"epoch": 1, "train_loss": 0.0, "val_loss": 0.0}
    summary = {
        "epochs_completed": int(history["epoch"].max()) if not history.empty else 0,
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "best_train_loss": float(best["train_loss"]),
        "feature_count": len(feature_cols),
        "label_mode": label_mode,
    }

    return model, history, summary


def predict_with_model(model: TrainedModel, features_df: pd.DataFrame) -> pd.DataFrame:
    X = features_df.reindex(columns=model.feature_columns).fillna(0.0)
    out = pd.DataFrame(index=features_df.index)

    if model.label_mode in {"event_classification", "risk_classification", "time_to_event_regression"}:
        m = model.task_models["target"]
        if model.label_mode == "time_to_event_regression":
            out["predicted_time_to_event_s"] = np.maximum(0, m.predict(X))
            out["predicted_event_type"] = np.where(out["predicted_time_to_event_s"] < 10, "event_soon", "normal")
            out["predicted_risk_level"] = np.where(out["predicted_time_to_event_s"] < 10, "high", "low")
            out["risk_score"] = np.clip(1 - out["predicted_time_to_event_s"] / 120, 0, 1)
        else:
            pred = m.predict(X)
            prob = m.predict_proba(X)
            classes = list(m.classes_)
            out["predicted_event_type"] = pred if model.label_mode == "event_classification" else "normal"
            out["predicted_risk_level"] = pred if model.label_mode == "risk_classification" else "low"
            out["risk_score"] = prob.max(axis=1)
            for i, cls in enumerate(classes):
                out[f"prob_{cls}"] = prob[:, i]
            out["predicted_time_to_event_s"] = np.maximum(0, 120 * (1 - out["risk_score"]))
    else:
        event_model = model.task_models["event"]
        risk_model = model.task_models["risk"]
        tte_model = model.task_models["tte"]

        event_pred = event_model.predict(X)
        risk_pred = risk_model.predict(X)
        event_prob = event_model.predict_proba(X)

        out["predicted_event_type"] = event_pred
        out["predicted_risk_level"] = risk_pred
        out["predicted_time_to_event_s"] = np.maximum(0, tte_model.predict(X))
        out["risk_score"] = event_prob.max(axis=1)

        for i, cls in enumerate(event_model.classes_):
            out[f"prob_{cls}"] = event_prob[:, i]

    out["predicted_event_location_m"] = features_df.get("event_location_m", pd.Series(0, index=features_df.index)).fillna(0)
    out["uncertainty_score"] = np.clip(1 - out.get("risk_score", 0.5), 0, 1)
    return out.reset_index(drop=True)
