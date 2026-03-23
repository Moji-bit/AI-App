from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "baseline_lstm": {"model_type": "LSTM", "hidden_dim": 128, "num_layers": 2, "dropout": 0.2},
    "baseline_gru": {"model_type": "GRU", "hidden_dim": 128, "num_layers": 2, "dropout": 0.2},
    "baseline_1dcnn": {"model_type": "1D CNN", "hidden_dim": 96, "num_layers": 3, "dropout": 0.15},
    "transformer_small": {"model_type": "Transformer Encoder", "d_model": 128, "num_layers": 2, "num_heads": 4, "dropout": 0.1},
    "transformer_multitask": {"model_type": "Multi-Task Transformer", "d_model": 192, "num_layers": 4, "num_heads": 6, "dropout": 0.1},
    "hybrid_cnn_lstm": {"model_type": "Hybrid CNN + LSTM", "hidden_dim": 160, "num_layers": 3, "dropout": 0.2},
    "hybrid_cnn_transformer": {"model_type": "Hybrid CNN + Transformer", "d_model": 192, "num_layers": 3, "num_heads": 6, "dropout": 0.15},
}


@dataclass
class ModelConfig:
    model_type: str = "LSTM"
    input_dim: int = 32
    hidden_dim: int = 128
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    sequence_length: int = 30
    forecast_horizon: int = 5
    output_heads: list[str] | None = None
    loss_weights: dict[str, float] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(data: str) -> "ModelConfig":
        obj = json.loads(data)
        return ModelConfig(**obj)


def config_from_preset(preset: str) -> ModelConfig:
    base = ModelConfig(output_heads=["event", "risk", "tte"], loss_weights={"event": 1.0, "risk": 0.5, "tte": 0.2})
    if preset in MODEL_PRESETS:
        d = asdict(base)
        d.update(MODEL_PRESETS[preset])
        return ModelConfig(**d)
    return base


def available_model_types() -> list[str]:
    return [
        "LSTM",
        "GRU",
        "1D CNN",
        "Transformer Encoder",
        "Multi-Task Transformer",
        "Hybrid CNN + LSTM",
        "Hybrid CNN + Transformer",
    ]
