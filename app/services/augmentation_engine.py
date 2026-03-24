from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


PRESETS: dict[str, dict[str, Any]] = {
    "normal traffic": {"event_type": "normal", "augmentation_strength": 0.25},
    "congestion": {"event_type": "congestion", "augmentation_strength": 0.45},
    "accident": {"event_type": "accident", "augmentation_strength": 0.6},
    "fire": {"event_type": "fire", "augmentation_strength": 0.7},
    "sensor fault": {"event_type": "sensor_fault", "augmentation_strength": 0.5},
    "winter weather": {"weather_type": "snow", "augmentation_strength": 0.55},
    "heavy rain": {"weather_type": "rain", "augmentation_strength": 0.5},
    "mixed disturbance": {"event_type": "accident", "weather_type": "storm", "augmentation_strength": 0.75},
}


@dataclass
class AugmentationConfig:
    target_scenarios: int = 1000
    augmentation_strength: float = 0.45
    noise_level: float = 0.03
    event_shift_range_s: int = 30
    missing_rate: float = 0.01
    outlier_rate: float = 0.002
    allowed_weather: list[str] = field(default_factory=lambda: ["clear", "rain", "snow", "fog", "storm"])
    class_balance_targets: dict[str, float] = field(
        default_factory=lambda: {
            "normal": 0.20,
            "congestion": 0.20,
            "accident": 0.20,
            "fire": 0.20,
            "sensor_fault": 0.20,
        }
    )
    seed: int = 42


CONTINUOUS_COLUMNS = [
    "speed_mean_kmh",
    "flow_veh_h",
    "occupancy_pct",
    "queue_length_m",
    "co_ppm",
    "no2_ppm",
    "pm25_ug_m3",
    "temp_c",
    "humidity_pct",
    "visibility_m",
    "air_velocity_mps",
    "pressure_hpa",
    "jet_fan_power_pct",
]

CLIP_RANGES = {
    "occupancy_pct": (0, 100),
    "visibility_m": (0, 3000),
    "jet_fan_power_pct": (0, 100),
    "speed_mean_kmh": (0, 160),
    "queue_length_m": (0, 10000),
}


class AugmentationEngine:
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def augment_scenario(
        self,
        scenario_metadata_row: pd.Series,
        scenario_timeseries: pd.DataFrame,
        scenario_ground_truth: pd.DataFrame,
        new_scenario_id: str,
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, float]:
        meta = scenario_metadata_row.copy()
        ts = scenario_timeseries.copy().sort_values("timestamp_s")
        gt = scenario_ground_truth.copy().sort_values("timestamp_s")

        meta["scenario_id"] = new_scenario_id
        meta["random_seed"] = int(self.rng.integers(0, 999999))

        shift = int(self.rng.integers(-self.config.event_shift_range_s, self.config.event_shift_range_s + 1))
        meta["event_start_s"] = max(0, float(meta.get("event_start_s", 0)) + shift)

        duration_scale = self.rng.uniform(0.8, 1.2)
        meta["event_duration_s"] = max(0, float(meta.get("event_duration_s", 0)) * duration_scale)
        meta["event_location_m"] = max(0, float(meta.get("event_location_m", 0)) + self.rng.normal(0, 35))
        meta["entry_flow_veh_h"] = max(0, float(meta.get("entry_flow_veh_h", 0)) * self.rng.uniform(0.8, 1.2))
        meta["heavy_vehicle_pct"] = float(np.clip(float(meta.get("heavy_vehicle_pct", 0)) + self.rng.normal(0, 2), 0, 100))
        meta["outside_temp_c"] = float(meta.get("outside_temp_c", 0)) + self.rng.normal(0, 2.5)
        meta["wind_speed_mps"] = max(0, float(meta.get("wind_speed_mps", 0)) + self.rng.normal(0, 1.0))
        meta["speed_limit_kmh"] = float(np.clip(float(meta.get("speed_limit_kmh", 80)) + self.rng.normal(0, 5), 30, 130))

        if self.config.allowed_weather:
            meta["weather_type"] = self.rng.choice(self.config.allowed_weather)

        event_type = str(meta.get("event_type", "normal"))
        severity = str(meta.get("event_severity", "low"))

        ts["scenario_id"] = new_scenario_id
        gt["scenario_id"] = new_scenario_id

        ts = self._augment_signals(ts)
        ts, gt = self._apply_event_aware_rules(ts, gt, event_type, severity, meta)

        gt = self._sync_ground_truth(gt, meta)
        quality_score = self._scenario_quality_score(ts, gt, meta)
        return meta, ts, gt, quality_score

    def _augment_signals(self, ts: pd.DataFrame) -> pd.DataFrame:
        out = ts.copy()
        n = len(out)
        x = np.linspace(0, 1, num=max(n, 2))

        for col in CONTINUOUS_COLUMNS:
            if col not in out.columns:
                continue
            series = pd.to_numeric(out[col], errors="coerce").interpolate().bfill().ffill()
            baseline = series.to_numpy(dtype=float)
            std = np.std(baseline) if np.std(baseline) > 0 else max(np.mean(np.abs(baseline)), 1.0)

            noise = self.rng.normal(0, std * self.config.noise_level, size=n)
            drift = (x - 0.5) * std * self.config.augmentation_strength * self.rng.uniform(-0.3, 0.3)
            scale = self.rng.uniform(1 - self.config.augmentation_strength * 0.2, 1 + self.config.augmentation_strength * 0.2)
            lag_steps = int(self.rng.integers(0, 3))

            augmented = baseline * scale + noise + drift
            if lag_steps > 0:
                augmented = np.roll(augmented, lag_steps)
                augmented[:lag_steps] = augmented[lag_steps]

            if n > 5:
                window = int(self.rng.integers(2, min(6, n)))
                smooth = pd.Series(augmented).rolling(window=window, min_periods=1, center=True).mean().to_numpy()
                mix = self.rng.uniform(0.25, 0.6)
                augmented = augmented * (1 - mix) + smooth * mix

            outlier_mask = self.rng.random(n) < self.config.outlier_rate
            augmented[outlier_mask] += self.rng.normal(0, std * 2.5, size=outlier_mask.sum())

            missing_mask = self.rng.random(n) < self.config.missing_rate
            augmented = augmented.astype(float)
            augmented[missing_mask] = np.nan

            out[col] = augmented

        for col, (low, high) in CLIP_RANGES.items():
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=low, upper=high)

        binary_cols = [
            "barrier_entry_state",
            "barrier_exit_state",
            "fire_alarm_state",
            "sos_calls_active",
            "emergency_mode_active",
            "sensor_fault_active",
            "fan_fault_active",
            "camera_fault_active",
        ]
        for col in binary_cols:
            if col in out.columns:
                vals = pd.to_numeric(out[col], errors="coerce").fillna(0)
                flip_prob = 0.01 + self.config.augmentation_strength * 0.02
                flips = self.rng.random(n) < flip_prob
                vals = vals.astype(int).to_numpy()
                vals[flips] = 1 - vals[flips]
                out[col] = vals

        if "timestamp_s" in out.columns:
            sample_jitter = self.rng.choice([0, 0, 0, 1, -1], size=n)
            out["timestamp_s"] = pd.to_numeric(out["timestamp_s"], errors="coerce") + sample_jitter
            out["timestamp_s"] = out["timestamp_s"].clip(lower=0)
            out = out.sort_values("timestamp_s").drop_duplicates(subset=["timestamp_s"], keep="first")

        return out

    def _apply_event_aware_rules(
        self,
        ts: pd.DataFrame,
        gt: pd.DataFrame,
        event_type: str,
        severity: str,
        meta: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        out_ts = ts.copy()
        out_gt = gt.copy()

        start = float(meta.get("event_start_s", 0))
        end = start + float(meta.get("event_duration_s", 0))
        event_mask = (pd.to_numeric(out_ts["timestamp_s"], errors="coerce") >= start) & (
            pd.to_numeric(out_ts["timestamp_s"], errors="coerce") <= end
        )

        sev_factor = {"low": 0.7, "medium": 1.0, "high": 1.3, "critical": 1.6}.get(severity, 1.0)

        if event_type == "accident":
            out_ts.loc[event_mask, "speed_mean_kmh"] *= max(0.2, 1 - 0.45 * sev_factor)
            out_ts.loc[event_mask, "queue_length_m"] *= 1 + 1.1 * sev_factor
            out_ts.loc[event_mask, "stopped_vehicle_count"] = (
                pd.to_numeric(out_ts.loc[event_mask, "stopped_vehicle_count"], errors="coerce").fillna(0) + self.rng.integers(1, 4)
            )
        elif event_type == "congestion":
            out_ts.loc[event_mask, "occupancy_pct"] = (
                pd.to_numeric(out_ts.loc[event_mask, "occupancy_pct"], errors="coerce").fillna(0) * (1.15 + 0.2 * sev_factor)
            )
            out_ts.loc[event_mask, "speed_mean_kmh"] *= max(0.25, 1 - 0.5 * sev_factor)
            out_ts.loc[event_mask, "queue_length_m"] *= 1 + 0.9 * sev_factor
        elif event_type == "fire":
            out_ts.loc[event_mask, "fire_alarm_state"] = 1
            out_ts.loc[event_mask, "visibility_m"] *= max(0.1, 1 - 0.65 * sev_factor)
            out_ts.loc[event_mask, "co_ppm"] *= 1 + 0.8 * sev_factor
            out_ts.loc[event_mask, "jet_fan_active_count"] = (
                pd.to_numeric(out_ts.loc[event_mask, "jet_fan_active_count"], errors="coerce").fillna(0) + 2
            )
            out_ts.loc[event_mask, "jet_fan_power_pct"] *= 1 + 0.45 * sev_factor
        elif event_type == "sensor_fault":
            fault_col = self.rng.choice(["sensor_fault_active", "fan_fault_active", "camera_fault_active"])
            out_ts.loc[event_mask, fault_col] = 1

        for col, (low, high) in CLIP_RANGES.items():
            if col in out_ts.columns:
                out_ts[col] = pd.to_numeric(out_ts[col], errors="coerce").clip(lower=low, upper=high)

        out_gt["timestamp_s"] = pd.to_numeric(out_gt["timestamp_s"], errors="coerce")
        out_gt["label_event_type"] = np.where(event_mask.reindex(out_gt.index, fill_value=False), event_type, "normal")
        out_gt["label_event_active"] = np.where(event_mask.reindex(out_gt.index, fill_value=False), 1, 0)

        risk = "low"
        if event_type in {"accident", "sensor_fault"}:
            risk = "medium" if severity in {"low", "medium"} else "high"
        if event_type == "fire":
            risk = "critical" if severity in {"high", "critical"} else "high"
        out_gt["label_risk_level"] = np.where(out_gt["label_event_active"] == 1, risk, "low")
        out_gt["label_event_location_m"] = np.where(
            out_gt["label_event_active"] == 1,
            float(meta.get("event_location_m", 0)),
            0,
        )

        return out_ts, out_gt

    def _sync_ground_truth(self, gt: pd.DataFrame, meta: pd.Series) -> pd.DataFrame:
        out = gt.copy().sort_values("timestamp_s")
        start = float(meta.get("event_start_s", 0))
        end = start + float(meta.get("event_duration_s", 0))
        t = pd.to_numeric(out["timestamp_s"], errors="coerce")
        out["label_phase"] = np.select(
            [t < start, (t >= start) & (t <= end), t > end],
            ["pre_event", "event_active", "recovery"],
            default="normal",
        )
        if str(meta.get("event_type", "normal")) == "normal":
            out["label_phase"] = "normal"
        return out

    def _scenario_quality_score(self, ts: pd.DataFrame, gt: pd.DataFrame, meta: pd.Series) -> float:
        score = 100.0
        nans = ts.isna().mean().mean()
        score -= min(25, nans * 300)

        if "occupancy_pct" in ts.columns:
            invalid_occ = ((pd.to_numeric(ts["occupancy_pct"], errors="coerce") < 0) | (pd.to_numeric(ts["occupancy_pct"], errors="coerce") > 100)).mean()
            score -= invalid_occ * 30

        mismatch = 0.0
        if "label_event_active" in gt.columns and "timestamp_s" in ts.columns:
            event_type = str(meta.get("event_type", "normal"))
            if event_type != "normal":
                mismatch = (pd.to_numeric(gt["label_event_active"], errors="coerce").fillna(0) == 0).mean() * 10
        score -= mismatch

        volatility_penalty = 0.0
        if "speed_mean_kmh" in ts.columns:
            diff = pd.to_numeric(ts["speed_mean_kmh"], errors="coerce").diff().abs().mean()
            if diff > 35:
                volatility_penalty += min(20, (diff - 35) * 0.8)
        score -= volatility_penalty

        return float(np.clip(score, 0, 100))


def _normalize_targets(targets: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.0) for v in targets.values())
    if total <= 0:
        return {"normal": 1.0}
    return {k: max(v, 0.0) / total for k, v in targets.items()}


def generate_augmented_dataset(
    scenario_metadata: pd.DataFrame,
    timeseries: pd.DataFrame,
    ground_truth: pd.DataFrame,
    config: AugmentationConfig,
) -> dict[str, pd.DataFrame]:
    engine = AugmentationEngine(config)

    meta = scenario_metadata.copy()
    ts = timeseries.copy()
    gt = ground_truth.copy()

    distribution = _normalize_targets(config.class_balance_targets)
    requested_by_class = {
        cls: int(round(config.target_scenarios * weight)) for cls, weight in distribution.items()
    }

    outputs_meta: list[pd.Series] = []
    outputs_ts: list[pd.DataFrame] = []
    outputs_gt: list[pd.DataFrame] = []

    class_groups = meta.groupby(meta["event_type"].astype(str).str.lower())

    generated = 0
    scenario_counter = 1
    while generated < config.target_scenarios:
        for event_class, target_count in requested_by_class.items():
            if generated >= config.target_scenarios:
                break
            if target_count <= 0:
                continue

            source_group = class_groups.get_group((event_class,)) if event_class in class_groups.groups else meta
            source_row = source_group.sample(1, random_state=int(engine.rng.integers(0, 999999))).iloc[0]
            sid = str(source_row["scenario_id"])

            source_ts = ts[ts["scenario_id"] == sid]
            source_gt = gt[gt["scenario_id"] == sid]
            if source_ts.empty or source_gt.empty:
                continue

            new_sid = f"AUG_{event_class[:3].upper()}_{scenario_counter:06d}"
            aug_meta, aug_ts, aug_gt, quality = engine.augment_scenario(source_row, source_ts, source_gt, new_sid)
            aug_meta["scenario_quality_score"] = quality
            outputs_meta.append(aug_meta)
            outputs_ts.append(aug_ts)
            outputs_gt.append(aug_gt)
            generated += 1
            scenario_counter += 1

            requested_by_class[event_class] -= 1

            if all(v <= 0 for v in requested_by_class.values()):
                requested_by_class = {
                    cls: max(1, int(round(config.target_scenarios * weight * 0.2)))
                    for cls, weight in distribution.items()
                }

    augmented_meta = pd.DataFrame(outputs_meta).reset_index(drop=True)
    augmented_ts = pd.concat(outputs_ts, ignore_index=True) if outputs_ts else pd.DataFrame()
    augmented_gt = pd.concat(outputs_gt, ignore_index=True) if outputs_gt else pd.DataFrame()

    return {
        "augmented_scenario_metadata": augmented_meta,
        "augmented_timeseries": augmented_ts,
        "augmented_ground_truth": augmented_gt,
    }
