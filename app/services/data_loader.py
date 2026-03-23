from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable

import pandas as pd


class DataLoaderError(ValueError):
    """Raised when CSV loading fails."""


def _read_csv(source: str | Path | BinaryIO | bytes, *, name: str) -> pd.DataFrame:
    try:
        if isinstance(source, (str, Path)):
            return pd.read_csv(source)
        if isinstance(source, bytes):
            return pd.read_csv(BytesIO(source))
        return pd.read_csv(source)
    except Exception as exc:  # pragma: no cover - pandas surface
        raise DataLoaderError(f"Could not load {name}: {exc}") from exc


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def load_tunnel_config(source: str | Path | BinaryIO | bytes) -> pd.DataFrame:
    return _normalize_columns(_read_csv(source, name="tunnel_config.csv"))


def load_scenario_metadata(source: str | Path | BinaryIO | bytes) -> pd.DataFrame:
    return _normalize_columns(_read_csv(source, name="scenario_metadata.csv"))


def load_timeseries(source: str | Path | BinaryIO | bytes) -> pd.DataFrame:
    return _normalize_columns(_read_csv(source, name="timeseries.csv"))


def load_ground_truth(source: str | Path | BinaryIO | bytes) -> pd.DataFrame:
    return _normalize_columns(_read_csv(source, name="ground_truth.csv"))


def load_all_sources(
    tunnel_config: str | Path | BinaryIO | bytes,
    scenario_metadata: str | Path | BinaryIO | bytes,
    timeseries: str | Path | BinaryIO | bytes,
    ground_truth: str | Path | BinaryIO | bytes,
) -> dict[str, pd.DataFrame]:
    return {
        "tunnel_config": load_tunnel_config(tunnel_config),
        "scenario_metadata": load_scenario_metadata(scenario_metadata),
        "timeseries": load_timeseries(timeseries),
        "ground_truth": load_ground_truth(ground_truth),
    }


def validate_non_empty_frames(frames: dict[str, pd.DataFrame], required: Iterable[str]) -> list[str]:
    errors: list[str] = []
    for key in required:
        if key not in frames:
            errors.append(f"Missing frame: {key}")
            continue
        if frames[key].empty:
            errors.append(f"Frame {key} is empty")
    return errors
