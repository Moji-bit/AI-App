from __future__ import annotations

import pandas as pd

from app.services.augmentation_engine import AugmentationConfig, generate_augmented_dataset
from app.services.data_merger import build_merged_dataset
from app.services.dataset_builder import DatasetBuildConfig, add_training_targets, build_windowed_training_dataset, train_val_test_split
from app.services.model_factory import ModelConfig
from app.services.schema_validator import validate_schema


def sample_frames() -> dict[str, pd.DataFrame]:
    tunnel = pd.DataFrame(
        [
            {
                "tunnel_id": "TUN_1",
                "tunnel_name": "Test",
                "tunnel_type": "road",
                "length_m": 1000,
                "width_m": 9,
                "clearance_height_m": 4.5,
                "gradient_pct": 1.0,
                "curvature_radius_m": 0,
                "cross_section_profile": "horseshoe",
                "tube_count": 1,
                "lanes_per_tube": 2,
                "traffic_mode": "bidirectional",
                "ventilation_type": "jet_fan",
                "jet_fan_count": 4,
                "camera_count": 2,
                "fire_detector_count": 4,
                "emergency_station_count": 2,
                "lighting_zones_count": 3,
                "segment_length_m": 100,
                "segment_count": 10,
            }
        ]
    )

    meta = pd.DataFrame(
        [
            {
                "scenario_id": "SCN_1",
                "run_id": "RUN_1",
                "tunnel_id": "TUN_1",
                "random_seed": 1,
                "simulation_start_ts": "2026-01-01T00:00:00Z",
                "simulation_duration_s": 10,
                "time_step_s": 2,
                "weather_type": "rain",
                "outside_temp_c": 10,
                "outside_humidity_pct": 50,
                "outside_pressure_hpa": 1013,
                "wind_direction": 90,
                "wind_speed_mps": 3,
                "daytime": "day",
                "traffic_demand_level": "high",
                "aadt": 10000,
                "heavy_vehicle_pct": 15,
                "speed_limit_kmh": 80,
                "entry_flow_veh_h": 900,
                "event_type": "accident",
                "event_start_s": 4,
                "event_duration_s": 4,
                "event_location_m": 500,
                "event_tube": 1,
                "event_lane": 1,
                "event_severity": "high",
            }
        ]
    )

    ts_rows, gt_rows = [], []
    for t in [0, 2, 4, 6, 8, 10]:
        ts_rows.append(
            {
                "scenario_id": "SCN_1",
                "timestamp_s": t,
                "speed_mean_kmh": 70,
                "flow_veh_h": 1000,
                "occupancy_pct": 45,
                "vehicle_count": 10,
                "queue_length_m": 5,
                "stopped_vehicle_count": 0,
                "heavy_vehicle_count": 1,
                "co_ppm": 1.0,
                "no2_ppm": 0.02,
                "pm25_ug_m3": 10,
                "temp_c": 20,
                "humidity_pct": 50,
                "visibility_m": 900,
                "air_velocity_mps": 2,
                "pressure_hpa": 1000,
                "jet_fan_active_count": 2,
                "jet_fan_power_pct": 40,
                "barrier_entry_state": 0,
                "barrier_exit_state": 0,
                "fire_alarm_state": 0,
                "lighting_mode": "normal",
                "camera_alarm_count": 0,
                "sos_calls_active": 0,
                "emergency_mode_active": 0,
                "sensor_fault_active": 0,
                "fan_fault_active": 0,
                "camera_fault_active": 0,
            }
        )
        gt_rows.append(
            {
                "scenario_id": "SCN_1",
                "timestamp_s": t,
                "label_event_type": "accident" if t >= 4 else "normal",
                "label_event_active": 1 if t >= 4 else 0,
                "label_risk_level": "high" if t >= 4 else "low",
                "label_phase": "event_active" if t >= 4 else "pre_event",
                "label_event_location_m": 500 if t >= 4 else 0,
            }
        )

    return {
        "tunnel_config": tunnel,
        "scenario_metadata": meta,
        "timeseries": pd.DataFrame(ts_rows),
        "ground_truth": pd.DataFrame(gt_rows),
    }


def test_end_to_end_build() -> None:
    frames = sample_frames()
    assert validate_schema(frames).schema_ok

    augmented = generate_augmented_dataset(
        frames["scenario_metadata"],
        frames["timeseries"],
        frames["ground_truth"],
        AugmentationConfig(target_scenarios=6, seed=3),
    )
    merged = build_merged_dataset(
        {
            "tunnel_config": frames["tunnel_config"],
            "scenario_metadata": augmented["augmented_scenario_metadata"],
            "timeseries": augmented["augmented_timeseries"],
            "ground_truth": augmented["augmented_ground_truth"],
        }
    )

    ds_cfg = DatasetBuildConfig(sequence_length=3, forecast_horizon=1, stride=1)
    windowed = build_windowed_training_dataset(merged, ds_cfg)
    windowed = add_training_targets(windowed, "multi_task")
    splits = train_val_test_split(windowed, ds_cfg)

    assert not windowed.empty
    assert all(k in splits for k in ["train", "val", "test"])


def test_model_json() -> None:
    cfg = ModelConfig(model_type="GRU", input_dim=16)
    loaded = ModelConfig.from_json(cfg.to_json())
    assert loaded.model_type == "GRU"
    assert loaded.input_dim == 16
