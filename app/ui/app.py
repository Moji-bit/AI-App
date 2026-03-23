from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui.pages import (
    analysis_evaluation,
    data_augmentation,
    data_import,
    dataset_validation,
    inspect_explainability,
    introduction,
    model_architecture,
    predict_inference,
    training,
    training_dataset_builder,
)


st.set_page_config(page_title="Tunnel Time-Series AI App", layout="wide")
st.title("Tunnel Time-Series AI App")

TAB_NAMES = [
    "1. Introduction",
    "2. Data Import",
    "3. Dataset Validation",
    "4. Data Augmentation",
    "5. Training Dataset Builder",
    "6. Model Architecture",
    "7. Training",
    "8. Inspect / Explainability",
    "9. Predict / Inference",
    "10. Analysis / Evaluation",
]

tabs = st.tabs(TAB_NAMES)

with tabs[0]:
    introduction.render()
with tabs[1]:
    data_import.render()
with tabs[2]:
    dataset_validation.render()
with tabs[3]:
    data_augmentation.render()
with tabs[4]:
    training_dataset_builder.render()
with tabs[5]:
    model_architecture.render()
with tabs[6]:
    training.render()
with tabs[7]:
    inspect_explainability.render()
with tabs[8]:
    predict_inference.render()
with tabs[9]:
    analysis_evaluation.render()
