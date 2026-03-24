from __future__ import annotations

from shiny import ui
from shinywidgets import output_widget

from .theme import app_styles


def build_layout() -> ui.Tag:
    return ui.page_navbar(
        ui.head_content(ui.tags.style(app_styles())),
        ui.div(
            {"class": "app-shell"},
            ui.div(
                {"class": "hero"},
                ui.h1("Tunnel AI Analytics Studio"),
                ui.p("End-to-End Workflow für Import, Validierung, Augmentation, Training, Inferenz und Evaluation in Shiny for Python."),
            ),
            ui.output_ui("kpi_cards"),
        ),
        ui.nav_panel(
            "Data Intake",
            ui.layout_columns(
                ui.card(
                    {"class": "card"},
                    ui.h4("1) Upload Rohdaten"),
                    ui.input_file("upload_tunnel", "tunnel_config.csv", accept=[".csv"]),
                    ui.input_file("upload_meta", "scenario_metadata.csv", accept=[".csv"]),
                    ui.input_file("upload_ts", "timeseries.csv", accept=[".csv"]),
                    ui.input_file("upload_gt", "ground_truth.csv", accept=[".csv"]),
                    ui.input_action_button("load_data", "Dateien laden", class_="btn-primary"),
                    ui.hr(),
                    ui.input_action_button("run_validation", "Schema validieren"),
                ),
                ui.card(
                    {"class": "card"},
                    ui.h4("2) Übersicht & Validierungsstatus"),
                    ui.output_text_verbatim("validation_summary"),
                    ui.output_data_frame("table_preview"),
                ),
                col_widths=[4, 8],
            ),
        ),
        ui.nav_panel(
            "Augmentation",
            ui.layout_columns(
                ui.card(
                    {"class": "card"},
                    ui.h4("Augmentation Konfiguration"),
                    ui.input_select("aug_preset", "Preset", choices=["custom", "normal traffic", "congestion", "accident", "fire", "sensor fault", "winter weather", "heavy rain", "mixed disturbance"]),
                    ui.input_numeric("target_scenarios", "Ziel-Szenarien", 1000, min=1, step=50),
                    ui.input_slider("aug_strength", "Augmentationsstärke", 0.0, 1.0, 0.45, step=0.01),
                    ui.input_slider("noise_level", "Noise", 0.0, 0.4, 0.03, step=0.005),
                    ui.input_numeric("event_shift", "Event Shift (s)", 30, min=0, step=1),
                    ui.input_slider("missing_rate", "Missing Rate", 0.0, 0.2, 0.01, step=0.001),
                    ui.input_slider("outlier_rate", "Outlier Rate", 0.0, 0.05, 0.002, step=0.001),
                    ui.input_numeric("aug_seed", "Random Seed", 42, min=0, step=1),
                    ui.input_text_area("class_targets", "Class-Balance (JSON)", '{"normal":0.2,"congestion":0.2,"accident":0.2,"fire":0.2,"sensor_fault":0.2}', rows=4),
                    ui.input_action_button("run_augmentation", "Augmentation starten", class_="btn-primary"),
                ),
                ui.card(
                    {"class": "card"},
                    ui.h4("Ergebnisse"),
                    ui.output_text_verbatim("augmentation_status"),
                    ui.output_data_frame("augmented_meta_table"),
                ),
                col_widths=[4, 8],
            ),
        ),
        ui.nav_panel(
            "Dataset Builder",
            ui.layout_columns(
                ui.card(
                    {"class": "card"},
                    ui.h4("Training Dataset"),
                    ui.input_checkbox("use_augmented", "Augmentierte Daten verwenden", True),
                    ui.input_numeric("seq_len", "Sequence Length", 30, min=3),
                    ui.input_numeric("forecast_horizon", "Forecast Horizon", 5, min=1),
                    ui.input_numeric("stride", "Stride", 5, min=1),
                    ui.input_select("label_mode", "Label Mode", choices=["event_classification", "risk_classification", "time_to_event_regression", "multi_task"]),
                    ui.input_slider("train_ratio", "Train Ratio", 0.1, 0.9, 0.7, step=0.05),
                    ui.input_slider("val_ratio", "Val Ratio", 0.05, 0.4, 0.15, step=0.05),
                    ui.input_action_button("build_dataset", "Training Dataset bauen", class_="btn-primary"),
                ),
                ui.card(
                    {"class": "card"},
                    ui.output_text_verbatim("dataset_status"),
                    output_widget("feature_plot"),
                    ui.output_data_frame("windowed_table"),
                ),
                col_widths=[4, 8],
            ),
        ),
        ui.nav_panel(
            "Model & Training",
            ui.layout_columns(
                ui.card(
                    {"class": "card"},
                    ui.h4("Modellkonfiguration"),
                    ui.input_select("model_preset", "Preset", choices=["baseline_lstm", "baseline_gru", "baseline_1dcnn", "transformer_small", "transformer_multitask", "hybrid_cnn_lstm", "hybrid_cnn_transformer"]),
                    ui.input_select("model_type", "Model", choices=["LSTM", "GRU", "1D CNN", "Transformer Encoder", "Multi-Task Transformer", "Hybrid CNN + LSTM", "Hybrid CNN + Transformer"]),
                    ui.input_numeric("input_dim", "Input Dim", 32, min=1),
                    ui.input_numeric("hidden_dim", "Hidden Dim", 128, min=4),
                    ui.input_numeric("d_model", "d_model", 128, min=8),
                    ui.input_numeric("num_layers", "Num Layers", 2, min=1),
                    ui.input_numeric("num_heads", "Num Heads", 4, min=1),
                    ui.input_slider("dropout", "Dropout", 0.0, 0.8, 0.2, step=0.01),
                    ui.input_text_area("loss_weights", "Loss Weights (JSON)", '{"event":1.0,"risk":0.5,"tte":0.2}', rows=3),
                    ui.input_action_button("save_model_cfg", "Konfiguration speichern"),
                    ui.hr(),
                    ui.h4("Training"),
                    ui.input_numeric("epochs", "Epochs", 40, min=1),
                    ui.input_numeric("batch_size", "Batch Size", 128, min=1),
                    ui.input_numeric("learning_rate", "Learning Rate", 0.001, min=0.00001),
                    ui.input_select("optimizer", "Optimizer", choices=["adam", "adamw", "sgd"]),
                    ui.input_select("scheduler", "Scheduler", choices=["none", "cosine", "step"]),
                    ui.input_checkbox("early_stopping", "Early Stopping", True),
                    ui.input_numeric("patience", "Patience", 8, min=1),
                    ui.input_numeric("train_seed", "Random Seed", 42, min=0),
                    ui.input_text_area("class_weights", "Class Weights (JSON)", '{"normal":1.0,"accident":2.0,"fire":3.0}', rows=3),
                    ui.input_action_button("run_training", "Training starten", class_="btn-primary"),
                ),
                ui.card(
                    {"class": "card"},
                    ui.output_text_verbatim("training_status"),
                    output_widget("training_plot"),
                    ui.output_data_frame("history_table"),
                ),
                col_widths=[4, 8],
            ),
        ),
        ui.nav_panel(
            "Inference & Evaluation",
            ui.layout_columns(
                ui.card(
                    {"class": "card"},
                    ui.h4("Inference"),
                    ui.input_select("pred_mode", "Mode", choices=["single scenario", "single window", "batch"]),
                    ui.input_numeric("window_index", "Window Index", 0, min=0),
                    ui.input_numeric("batch_size_pred", "Batch Size", 64, min=1),
                    ui.input_action_button("run_inference", "Inference starten", class_="btn-primary"),
                ),
                ui.card(
                    {"class": "card"},
                    ui.output_text_verbatim("eval_status"),
                    output_widget("cm_plot"),
                    output_widget("importance_plot"),
                    ui.output_data_frame("prediction_table"),
                ),
                col_widths=[4, 8],
            ),
        ),
        ui.nav_panel(
            "Export",
            ui.card(
                {"class": "card"},
                ui.h4("Dateiexport"),
                ui.input_text("export_dir", "Export-Verzeichnis", "exports"),
                ui.input_action_button("export_files", "Artefakte exportieren", class_="btn-primary"),
                ui.output_text_verbatim("export_status"),
            ),
        ),
        title="Tunnel AI App",
        id="main_nav",
    )
