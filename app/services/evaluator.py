from __future__ import annotations

import numpy as np
import pandas as pd


RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    labels = sorted(set(y_true.astype(str)) | set(y_pred.astype(str)))
    tp = sum((y_true.astype(str) == y_pred.astype(str)).astype(int))
    accuracy = _safe_div(tp, len(y_true))

    precisions, recalls, f1s = [], [], []
    for label in labels:
        t = (y_true.astype(str) == label)
        p = (y_pred.astype(str) == label)
        tp_l = int((t & p).sum())
        fp_l = int((~t & p).sum())
        fn_l = int((t & ~p).sum())
        precision = _safe_div(tp_l, tp_l + fp_l)
        recall = _safe_div(tp_l, tp_l + fn_l)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(np.mean(precisions) if precisions else 0.0),
        "recall_macro": float(np.mean(recalls) if recalls else 0.0),
        "f1_macro": float(np.mean(f1s) if f1s else 0.0),
    }


def confusion_matrix_df(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    labels = sorted(set(y_true.astype(str)) | set(y_pred.astype(str)))
    matrix = pd.crosstab(y_true.astype(str), y_pred.astype(str), rownames=["actual"], colnames=["predicted"], dropna=False)
    return matrix.reindex(index=labels, columns=labels, fill_value=0)


def compute_operational_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df.get("target_event_type", pd.Series(dtype=str)).astype(str)
    y_pred = df.get("predicted_event_type", pd.Series(dtype=str)).astype(str)

    event_true = y_true != "normal"
    event_pred = y_pred != "normal"

    fp = int((~event_true & event_pred).sum())
    fn = int((event_true & ~event_pred).sum())
    tp = int((event_true & event_pred).sum())
    tn = int((~event_true & ~event_pred).sum())

    false_alarm_rate = _safe_div(fp, fp + tn)
    missed_event_rate = _safe_div(fn, fn + tp)

    lead_time = 0.0
    if "target_time_to_event_s" in df.columns and "predicted_time_to_event_s" in df.columns:
        diff = pd.to_numeric(df["target_time_to_event_s"], errors="coerce") - pd.to_numeric(df["predicted_time_to_event_s"], errors="coerce")
        lead_time = float(diff.mean(skipna=True))

    robustness = 1.0
    if "sensor_fault_active_mean" in df.columns:
        fault_idx = pd.to_numeric(df["sensor_fault_active_mean"], errors="coerce") > 0.1
        if fault_idx.any():
            robustness = _safe_div(int((y_true[fault_idx] == y_pred[fault_idx]).sum()), int(fault_idx.sum()))

    return {
        "false_alarm_rate": float(false_alarm_rate),
        "missed_event_rate": float(missed_event_rate),
        "lead_time_s": float(lead_time),
        "robustness_under_sensor_faults": float(robustness),
    }
