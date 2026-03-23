from __future__ import annotations

import pandas as pd


def build_merged_dataset(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    metadata = frames["scenario_metadata"].copy()
    ts = frames["timeseries"].copy()
    gt = frames["ground_truth"].copy()
    tunnel = frames["tunnel_config"].copy()

    ts["timestamp_s"] = pd.to_numeric(ts["timestamp_s"], errors="coerce")
    gt["timestamp_s"] = pd.to_numeric(gt["timestamp_s"], errors="coerce")

    merged = ts.merge(
        gt,
        on=["scenario_id", "timestamp_s"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_gt"),
    )
    merged = merged.merge(metadata, on="scenario_id", how="left", validate="many_to_one")
    merged = merged.merge(tunnel, on="tunnel_id", how="left", validate="many_to_one", suffixes=("", "_tunnel"))

    merged = merged.sort_values(["scenario_id", "timestamp_s"]).reset_index(drop=True)
    return merged
