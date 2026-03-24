from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.express as px
from shiny import reactive, render, ui
from shinywidgets import render_widget

from app.services.augmentation_engine import AugmentationConfig, generate_augmented_dataset
from app.services.data_loader import load_ground_truth, load_scenario_metadata, load_timeseries, load_tunnel_config
from app.services.data_merger import build_merged_dataset
from app.services.dataset_builder import DatasetBuildConfig, add_training_targets, build_windowed_training_dataset, train_val_test_split
from app.services.evaluator import confusion_matrix_df, evaluate_predictions
from app.services.explainability import feature_importance_df
from app.services.export_service import export_augmented_files
from app.services.model_factory import ModelConfig, config_from_preset
from app.services.schema_validator import validate_schema
from app.services.trainer import TrainingConfig, predict_with_model, train_model


def _file_path(fileinfo: list[dict[str, Any]] | None) -> str | None:
    return fileinfo[0]["datapath"] if fileinfo else None


def register_handlers(input, output, session) -> None:
    state = reactive.value(
        {
            "frames": None,
            "validation": None,
            "augmented": None,
            "merged": None,
            "windowed": None,
            "splits": None,
            "model_config": None,
            "trained_model": None,
            "history": None,
            "predictions": None,
            "summary": None,
        }
    )

    def _get() -> dict[str, Any]:
        return state.get()

    def _set(**kwargs) -> None:
        cur = _get().copy()
        cur.update(kwargs)
        state.set(cur)

    @reactive.effect
    @reactive.event(input.model_preset)
    def _preset_sync():
        cfg = config_from_preset(input.model_preset())
        ui.update_select("model_type", selected=cfg.model_type)
        ui.update_numeric("hidden_dim", value=cfg.hidden_dim)
        ui.update_numeric("d_model", value=cfg.d_model)
        ui.update_numeric("num_layers", value=cfg.num_layers)
        ui.update_numeric("num_heads", value=cfg.num_heads)
        ui.update_slider("dropout", value=cfg.dropout)

    @reactive.effect
    @reactive.event(input.load_data)
    def _load_data():
        paths = {
            "tunnel": _file_path(input.upload_tunnel()),
            "meta": _file_path(input.upload_meta()),
            "ts": _file_path(input.upload_ts()),
            "gt": _file_path(input.upload_gt()),
        }
        if not all(paths.values()):
            ui.notification_show("Bitte alle vier CSV-Dateien hochladen.", type="warning")
            return
        frames = {
            "tunnel_config": load_tunnel_config(paths["tunnel"]),
            "scenario_metadata": load_scenario_metadata(paths["meta"]),
            "timeseries": load_timeseries(paths["ts"]),
            "ground_truth": load_ground_truth(paths["gt"]),
        }
        _set(frames=frames)
        ui.notification_show("Daten erfolgreich geladen.", type="message")

    @reactive.effect
    @reactive.event(input.run_validation)
    def _validate():
        s = _get()
        if not s["frames"]:
            ui.notification_show("Zuerst Daten laden.", type="warning")
            return
        rep = validate_schema(s["frames"])
        _set(validation=rep)
        ui.notification_show("Validierung abgeschlossen.", type="message" if rep.schema_ok else "warning")

    @reactive.effect
    @reactive.event(input.run_augmentation)
    def _augment():
        s = _get()
        if not s["frames"]:
            ui.notification_show("Keine Daten geladen.", type="warning")
            return
        try:
            class_targets = {str(k): float(v) for k, v in json.loads(input.class_targets()).items()}
        except Exception as exc:
            ui.notification_show(f"Ungültiges Class-Balance JSON: {exc}", type="error")
            return

        cfg = AugmentationConfig(
            target_scenarios=int(input.target_scenarios()),
            augmentation_strength=float(input.aug_strength()),
            noise_level=float(input.noise_level()),
            event_shift_range_s=int(input.event_shift()),
            missing_rate=float(input.missing_rate()),
            outlier_rate=float(input.outlier_rate()),
            seed=int(input.aug_seed()),
            class_balance_targets=class_targets,
        )
        augmented = generate_augmented_dataset(
            s["frames"]["scenario_metadata"],
            s["frames"]["timeseries"],
            s["frames"]["ground_truth"],
            cfg,
        )
        _set(augmented=augmented)
        ui.notification_show("Augmentation erfolgreich abgeschlossen.", type="message")

    @reactive.effect
    @reactive.event(input.build_dataset)
    def _build_dataset():
        s = _get()
        frames = s["frames"]
        if not frames:
            ui.notification_show("Keine Daten geladen.", type="warning")
            return

        source = frames.copy()
        if input.use_augmented() and s["augmented"]:
            source = {
                "tunnel_config": frames["tunnel_config"],
                "scenario_metadata": s["augmented"]["augmented_scenario_metadata"],
                "timeseries": s["augmented"]["augmented_timeseries"],
                "ground_truth": s["augmented"]["augmented_ground_truth"],
            }

        cfg = DatasetBuildConfig(
            sequence_length=int(input.seq_len()),
            forecast_horizon=int(input.forecast_horizon()),
            stride=int(input.stride()),
            label_mode=input.label_mode(),
            train_ratio=float(input.train_ratio()),
            val_ratio=float(input.val_ratio()),
            test_ratio=float(1.0 - input.train_ratio() - input.val_ratio()),
        )
        try:
            merged = build_merged_dataset(source)
            windowed = build_windowed_training_dataset(merged, cfg)
            windowed = add_training_targets(windowed, input.label_mode())
            splits = train_val_test_split(windowed, cfg)
        except Exception as exc:
            ui.notification_show(f"Dataset-Erstellung fehlgeschlagen: {exc}", type="error")
            return

        _set(merged=merged, windowed=windowed, splits=splits)
        ui.notification_show("Training-Dataset wurde erstellt.", type="message")

    @reactive.effect
    @reactive.event(input.save_model_cfg)
    def _save_cfg():
        try:
            loss_weights = {str(k): float(v) for k, v in json.loads(input.loss_weights()).items()}
        except Exception as exc:
            ui.notification_show(f"Ungültiges Loss-JSON: {exc}", type="error")
            return
        cfg = ModelConfig(
            model_type=input.model_type(),
            input_dim=int(input.input_dim()),
            hidden_dim=int(input.hidden_dim()),
            d_model=int(input.d_model()),
            num_layers=int(input.num_layers()),
            num_heads=int(input.num_heads()),
            dropout=float(input.dropout()),
            sequence_length=int(input.seq_len()),
            forecast_horizon=int(input.forecast_horizon()),
            output_heads=["event", "risk", "tte"],
            loss_weights=loss_weights,
        )
        _set(model_config=cfg)
        ui.notification_show("Model-Konfiguration gespeichert.", type="message")

    @reactive.effect
    @reactive.event(input.run_training)
    def _train():
        s = _get()
        if s["windowed"] is None or s["windowed"].empty:
            ui.notification_show("Bitte zuerst Dataset bauen.", type="warning")
            return
        if s["model_config"] is None:
            ui.notification_show("Bitte zuerst Modellkonfiguration speichern.", type="warning")
            return
        try:
            class_weights = {str(k): float(v) for k, v in json.loads(input.class_weights()).items()}
        except Exception as exc:
            ui.notification_show(f"Ungültiges Class-Weights JSON: {exc}", type="error")
            return

        t_cfg = TrainingConfig(
            epochs=int(input.epochs()),
            batch_size=int(input.batch_size()),
            learning_rate=float(input.learning_rate()),
            optimizer=input.optimizer(),
            scheduler=input.scheduler(),
            early_stopping=bool(input.early_stopping()),
            patience=int(input.patience()),
            random_seed=int(input.train_seed()),
            class_weights=class_weights,
        )
        label_mode = "multi_task" if "target_event" in s["windowed"].columns else "target"
        train_df = s["splits"]["train"] if s["splits"] else s["windowed"]
        val_df = s["splits"]["val"] if s["splits"] else s["windowed"]
        model, history, summary = train_model(train_df, val_df, s["model_config"], t_cfg, label_mode)
        _set(trained_model=model, history=history, summary=summary)
        ui.notification_show("Training abgeschlossen.", type="message")

    @reactive.effect
    @reactive.event(input.run_inference)
    def _infer():
        s = _get()
        if s["trained_model"] is None or s["windowed"] is None or s["windowed"].empty:
            ui.notification_show("Bitte zuerst Dataset bauen und trainieren.", type="warning")
            return

        w = s["windowed"]
        mode = input.pred_mode()
        if mode == "single scenario":
            sid = w["scenario_id"].astype(str).iloc[0]
            features = w[w["scenario_id"].astype(str) == sid].head(1)
        elif mode == "single window":
            idx = int(max(0, min(input.window_index(), len(w) - 1)))
            features = w.iloc[[idx]]
        else:
            size = int(max(1, min(input.batch_size_pred(), len(w))))
            features = w.head(size)

        pred = predict_with_model(s["trained_model"], features)
        out = features.reset_index(drop=True).join(pred)
        _set(predictions=out)
        ui.notification_show("Inference abgeschlossen.", type="message")

    @reactive.effect
    @reactive.event(input.export_files)
    def _export():
        s = _get()
        if s["frames"] is None:
            ui.notification_show("Keine Daten geladen.", type="warning")
            return

        if s["augmented"] is not None:
            files = export_augmented_files(
                input.export_dir(),
                s["augmented"]["augmented_scenario_metadata"],
                s["augmented"]["augmented_timeseries"],
                s["augmented"]["augmented_ground_truth"],
                augmented_tunnel_config=s["frames"]["tunnel_config"],
                merged_training_dataset=s["merged"],
            )
            _set(summary={k: str(v) for k, v in files.items()})
            ui.notification_show("Export abgeschlossen.", type="message")

    @output
    @render.ui
    def kpi_cards():
        s = _get()
        frames = s["frames"]
        windowed = s["windowed"]
        preds = s["predictions"]
        cards = [
            ("Scenarios", len(frames["scenario_metadata"]) if frames else 0),
            ("Time-Series Rows", len(frames["timeseries"]) if frames else 0),
            ("Windowed Samples", len(windowed) if isinstance(windowed, pd.DataFrame) else 0),
            ("Predictions", len(preds) if isinstance(preds, pd.DataFrame) else 0),
        ]
        return ui.div(
            {"class": "kpis"},
            *[
                ui.div({"class": "kpi"}, ui.div({"class": "label"}, label), ui.div({"class": "value"}, str(value)))
                for label, value in cards
            ],
        )

    @output
    @render.text
    def validation_summary():
        rep = _get()["validation"]
        if rep is None:
            return "Noch keine Validierung durchgeführt."
        return json.dumps(rep.to_dict(), indent=2, ensure_ascii=False)

    @output
    @render.data_frame
    def table_preview():
        frames = _get()["frames"]
        if not frames:
            return render.DataGrid(pd.DataFrame({"info": ["Noch keine Daten geladen"]}))
        sample = frames["scenario_metadata"].head(50)
        return render.DataGrid(sample)

    @output
    @render.text
    def augmentation_status():
        aug = _get()["augmented"]
        if not aug:
            return "Keine Augmentation vorhanden."
        return f"Augmentierte Szenarien: {len(aug['augmented_scenario_metadata'])}"

    @output
    @render.data_frame
    def augmented_meta_table():
        aug = _get()["augmented"]
        if not aug:
            return render.DataGrid(pd.DataFrame())
        return render.DataGrid(aug["augmented_scenario_metadata"].head(100))

    @output
    @render.text
    def dataset_status():
        s = _get()
        if s["windowed"] is None:
            return "Noch kein Windowed-Dataset gebaut."
        split_sizes = {k: len(v) for k, v in (s["splits"] or {}).items()}
        return f"Windowed Samples: {len(s['windowed'])}\nSplits: {split_sizes}"

    @output
    @render_widget
    def feature_plot():
        windowed = _get()["windowed"]
        if windowed is None or windowed.empty:
            return px.scatter(title="Noch keine Features verfügbar")
        numeric = windowed.select_dtypes(include="number")
        if numeric.empty:
            return px.scatter(title="Keine numerischen Features")
        means = numeric.mean().sort_values(ascending=False).head(15)
        df = pd.DataFrame({"feature": means.index, "value": means.values})
        return px.bar(df, x="feature", y="value", title="Top Feature-Mittelwerte")

    @output
    @render.data_frame
    def windowed_table():
        w = _get()["windowed"]
        return render.DataGrid(w.head(100) if isinstance(w, pd.DataFrame) else pd.DataFrame())

    @output
    @render.text
    def training_status():
        s = _get()
        if not s["summary"]:
            return "Training wurde noch nicht ausgeführt."
        return json.dumps(s["summary"], indent=2, ensure_ascii=False)

    @output
    @render_widget
    def training_plot():
        history = _get()["history"]
        if history is None or history.empty:
            return px.line(title="Noch keine Trainingshistorie")
        return px.line(history, x="epoch", y=["train_loss", "val_loss"], title="Training vs Validation Loss")

    @output
    @render.data_frame
    def history_table():
        history = _get()["history"]
        return render.DataGrid(history if isinstance(history, pd.DataFrame) else pd.DataFrame())

    @output
    @render.text
    def eval_status():
        preds = _get()["predictions"]
        if preds is None or preds.empty:
            return "Keine Predictions verfügbar."
        if "target_event_type" not in preds.columns:
            return "Predictions enthalten keine Ground-Truth-Spalte 'target_event_type'."
        metrics = evaluate_predictions(preds["target_event_type"], preds["predicted_event_type"])
        return json.dumps(metrics, indent=2, ensure_ascii=False)

    @output
    @render_widget
    def cm_plot():
        preds = _get()["predictions"]
        if preds is None or preds.empty or "target_event_type" not in preds.columns:
            return px.imshow(pd.DataFrame([[0]], columns=["no_data"], index=["no_data"]), title="Confusion Matrix")
        cm = confusion_matrix_df(preds["target_event_type"], preds["predicted_event_type"])
        return px.imshow(cm, text_auto=True, title="Confusion Matrix")

    @output
    @render_widget
    def importance_plot():
        model = _get()["trained_model"]
        if model is None:
            return px.bar(title="Keine Feature Importance verfügbar")
        fi = feature_importance_df(model).head(20)
        return px.bar(fi, x="feature", y="importance", title="Feature Importance")

    @output
    @render.data_frame
    def prediction_table():
        preds = _get()["predictions"]
        return render.DataGrid(preds if isinstance(preds, pd.DataFrame) else pd.DataFrame())

    @output
    @render.text
    def export_status():
        summary = _get()["summary"]
        if not summary:
            return "Noch kein Export durchgeführt."
        return json.dumps(summary, indent=2, ensure_ascii=False)
