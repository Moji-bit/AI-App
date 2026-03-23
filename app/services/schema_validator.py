from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "tunnel_config": [
        "tunnel_id",
        "tunnel_name",
        "tunnel_type",
        "length_m",
        "width_m",
        "clearance_height_m",
        "gradient_pct",
        "curvature_radius_m",
        "cross_section_profile",
        "tube_count",
        "lanes_per_tube",
        "traffic_mode",
        "ventilation_type",
        "jet_fan_count",
        "camera_count",
        "fire_detector_count",
        "emergency_station_count",
        "lighting_zones_count",
        "segment_length_m",
        "segment_count",
    ],
    "scenario_metadata": [
        "scenario_id",
        "run_id",
        "tunnel_id",
        "random_seed",
        "simulation_start_ts",
        "simulation_duration_s",
        "time_step_s",
        "weather_type",
        "outside_temp_c",
        "outside_humidity_pct",
        "outside_pressure_hpa",
        "wind_direction",
        "wind_speed_mps",
        "daytime",
        "traffic_demand_level",
        "aadt",
        "heavy_vehicle_pct",
        "speed_limit_kmh",
        "entry_flow_veh_h",
        "event_type",
        "event_start_s",
        "event_duration_s",
        "event_location_m",
        "event_tube",
        "event_lane",
        "event_severity",
    ],
    "timeseries": [
        "scenario_id",
        "timestamp_s",
        "speed_mean_kmh",
        "flow_veh_h",
        "occupancy_pct",
        "vehicle_count",
        "queue_length_m",
        "stopped_vehicle_count",
        "heavy_vehicle_count",
        "co_ppm",
        "no2_ppm",
        "pm25_ug_m3",
        "temp_c",
        "humidity_pct",
        "visibility_m",
        "air_velocity_mps",
        "pressure_hpa",
        "jet_fan_active_count",
        "jet_fan_power_pct",
        "barrier_entry_state",
        "barrier_exit_state",
        "fire_alarm_state",
        "lighting_mode",
        "camera_alarm_count",
        "sos_calls_active",
        "emergency_mode_active",
        "sensor_fault_active",
        "fan_fault_active",
        "camera_fault_active",
    ],
    "ground_truth": [
        "scenario_id",
        "timestamp_s",
        "label_event_type",
        "label_event_active",
        "label_risk_level",
        "label_phase",
        "label_event_location_m",
    ],
}

NUMERIC_COLUMNS: dict[str, list[str]] = {
    "tunnel_config": [
        "length_m",
        "width_m",
        "clearance_height_m",
        "gradient_pct",
        "curvature_radius_m",
        "tube_count",
        "lanes_per_tube",
        "jet_fan_count",
        "camera_count",
        "fire_detector_count",
        "emergency_station_count",
        "lighting_zones_count",
        "segment_length_m",
        "segment_count",
    ],
    "scenario_metadata": [
        "random_seed",
        "simulation_duration_s",
        "time_step_s",
        "outside_temp_c",
        "outside_humidity_pct",
        "outside_pressure_hpa",
        "wind_direction",
        "wind_speed_mps",
        "aadt",
        "heavy_vehicle_pct",
        "speed_limit_kmh",
        "entry_flow_veh_h",
        "event_start_s",
        "event_duration_s",
        "event_location_m",
        "event_tube",
        "event_lane",
    ],
    "timeseries": [
        "timestamp_s",
        "speed_mean_kmh",
        "flow_veh_h",
        "occupancy_pct",
        "vehicle_count",
        "queue_length_m",
        "stopped_vehicle_count",
        "heavy_vehicle_count",
        "co_ppm",
        "no2_ppm",
        "pm25_ug_m3",
        "temp_c",
        "humidity_pct",
        "visibility_m",
        "air_velocity_mps",
        "pressure_hpa",
        "jet_fan_active_count",
        "jet_fan_power_pct",
        "barrier_entry_state",
        "barrier_exit_state",
        "fire_alarm_state",
        "camera_alarm_count",
        "sos_calls_active",
        "emergency_mode_active",
        "sensor_fault_active",
        "fan_fault_active",
        "camera_fault_active",
    ],
    "ground_truth": ["timestamp_s", "label_event_active", "label_event_location_m"],
}

ALLOWED_CATEGORICAL = {
    "weather_type": {"clear", "rain", "snow", "fog", "storm"},
    "event_type": {"normal", "congestion", "accident", "fire", "sensor_fault"},
    "event_severity": {"low", "medium", "high", "critical"},
}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_values: dict[str, dict[str, int]] = field(default_factory=dict)
    schema_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "missing_values": self.missing_values,
            "schema_ok": self.schema_ok,
        }


def _check_required_columns(name: str, df: pd.DataFrame, report: ValidationReport) -> None:
    missing = sorted(set(REQUIRED_COLUMNS[name]).difference(df.columns))
    if missing:
        report.errors.append(f"{name}: missing required columns: {', '.join(missing)}")


def _check_numeric_types(name: str, df: pd.DataFrame, report: ValidationReport) -> None:
    for column in NUMERIC_COLUMNS[name]:
        if column not in df.columns:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        bad = int(converted.isna().sum() - df[column].isna().sum())
        if bad > 0:
            report.errors.append(f"{name}.{column}: {bad} values are not numeric")


def _check_missing_values(name: str, df: pd.DataFrame, report: ValidationReport) -> None:
    missing = {col: int(cnt) for col, cnt in df.isna().sum().items() if int(cnt) > 0}
    if missing:
        report.missing_values[name] = missing
        report.warnings.append(f"{name}: missing values detected")


def _check_plausibility(frames: dict[str, pd.DataFrame], report: ValidationReport) -> None:
    ts = frames["timeseries"]
    checks = [
        ("occupancy_pct", 0, 100),
        ("jet_fan_power_pct", 0, 100),
        ("visibility_m", 0, np.inf),
    ]
    for col, low, high in checks:
        if col not in ts.columns:
            continue
        vals = pd.to_numeric(ts[col], errors="coerce")
        invalid = vals[(vals < low) | (vals > high)]
        if not invalid.empty:
            report.warnings.append(f"timeseries.{col}: {len(invalid)} values outside plausible range [{low}, {high}]")


def validate_schema(frames: dict[str, pd.DataFrame]) -> ValidationReport:
    report = ValidationReport()
    for name, required_cols in REQUIRED_COLUMNS.items():
        if name not in frames:
            report.errors.append(f"Missing required file: {name}")
            continue
        df = frames[name]
        _check_required_columns(name, df, report)
        _check_numeric_types(name, df, report)
        _check_missing_values(name, df, report)

        if name == "scenario_metadata" and "scenario_id" in df.columns:
            if df["scenario_id"].duplicated().any():
                report.errors.append("scenario_metadata.scenario_id contains duplicates")

    if all(k in frames for k in ["timeseries", "ground_truth", "scenario_metadata", "tunnel_config"]):
        _check_plausibility(frames, report)
        consistency_errors, consistency_warnings = validate_cross_file_consistency(frames)
        report.errors.extend(consistency_errors)
        report.warnings.extend(consistency_warnings)

    report.schema_ok = len(report.errors) == 0
    return report


def validate_cross_file_consistency(frames: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    meta = frames["scenario_metadata"]
    ts = frames["timeseries"]
    gt = frames["ground_truth"]
    tun = frames["tunnel_config"]

    meta_ids = set(meta["scenario_id"].astype(str))
    ts_ids = set(ts["scenario_id"].astype(str))
    gt_ids = set(gt["scenario_id"].astype(str))

    if ts_ids - meta_ids:
        errors.append(f"timeseries has unknown scenario_id values: {len(ts_ids - meta_ids)}")
    if gt_ids - meta_ids:
        errors.append(f"ground_truth has unknown scenario_id values: {len(gt_ids - meta_ids)}")

    tunnel_ids = set(tun["tunnel_id"].astype(str))
    meta_tunnel_ids = set(meta["tunnel_id"].astype(str))
    if meta_tunnel_ids - tunnel_ids:
        errors.append(f"scenario_metadata references unknown tunnel_id values: {len(meta_tunnel_ids - tunnel_ids)}")

    ts_index = set(zip(ts["scenario_id"].astype(str), pd.to_numeric(ts["timestamp_s"], errors="coerce")))
    gt_index = set(zip(gt["scenario_id"].astype(str), pd.to_numeric(gt["timestamp_s"], errors="coerce")))
    if ts_index != gt_index:
        only_ts = len(ts_index - gt_index)
        only_gt = len(gt_index - ts_index)
        errors.append(f"timestamp mismatch between timeseries and ground_truth (timeseries_only={only_ts}, ground_truth_only={only_gt})")

    for state_col in [
        "barrier_entry_state",
        "barrier_exit_state",
        "fire_alarm_state",
        "sos_calls_active",
        "emergency_mode_active",
        "sensor_fault_active",
        "fan_fault_active",
        "camera_fault_active",
    ]:
        if state_col in ts.columns:
            vals = set(pd.to_numeric(ts[state_col], errors="coerce").dropna().astype(int).unique())
            if not vals.issubset({0, 1}):
                warnings.append(f"timeseries.{state_col} contains non-binary values: {sorted(vals)}")

    for cat_col, allowed in ALLOWED_CATEGORICAL.items():
        if cat_col in meta.columns:
            values = set(meta[cat_col].astype(str).str.lower().unique())
            unknown = values - allowed
            if unknown:
                warnings.append(f"scenario_metadata.{cat_col} has out-of-list values: {sorted(unknown)}")

    return errors, warnings
