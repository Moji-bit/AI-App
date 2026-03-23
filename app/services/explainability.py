from __future__ import annotations

import numpy as np
import pandas as pd

from .trainer import TrainedModel


def feature_importance_df(model: TrainedModel) -> pd.DataFrame:
    items = sorted(model.feature_importance.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(items, columns=["feature", "importance"])


def attention_proxy(window_df: pd.DataFrame) -> pd.DataFrame:
    numeric = window_df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame(columns=["feature", "attention_weight"])
    weights = numeric.abs().mean()
    weights = weights / weights.sum() if weights.sum() > 0 else weights
    return pd.DataFrame({"feature": weights.index, "attention_weight": weights.values}).sort_values("attention_weight", ascending=False)


def error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if "target_event_type" not in df.columns or "predicted_event_type" not in df.columns:
        return pd.DataFrame(columns=["type", "count"])

    fp = ((df["target_event_type"].astype(str) == "normal") & (df["predicted_event_type"].astype(str) != "normal")).sum()
    fn = ((df["target_event_type"].astype(str) != "normal") & (df["predicted_event_type"].astype(str) == "normal")).sum()
    tp = ((df["target_event_type"].astype(str) != "normal") & (df["predicted_event_type"].astype(str) != "normal")).sum()

    return pd.DataFrame({"type": ["false_positive", "false_negative", "true_positive"], "count": [int(fp), int(fn), int(tp)]})
