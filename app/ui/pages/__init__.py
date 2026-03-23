from __future__ import annotations

from importlib import import_module
from types import ModuleType

PAGE_REGISTRY: list[tuple[str, str]] = [
    ("1. Introduction", "introduction"),
    ("2. Data Import", "data_import"),
    ("3. Dataset Validation", "dataset_validation"),
    ("4. Data Augmentation", "data_augmentation"),
    ("5. Training Dataset Builder", "training_dataset_builder"),
    ("6. Model Architecture", "model_architecture"),
    ("7. Training", "training"),
    ("8. Inspect / Explainability", "inspect_explainability"),
    ("9. Predict / Inference", "predict_inference"),
    ("10. Analysis / Evaluation", "analysis_evaluation"),
]


def load_pages(package: str = __name__) -> list[tuple[str, ModuleType]]:
    loaded: list[tuple[str, ModuleType]] = []
    for label, module_name in PAGE_REGISTRY:
        module = import_module(f"{package}.{module_name}")
        loaded.append((label, module))
    return loaded


__all__ = ["PAGE_REGISTRY", "load_pages"]
