from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shiny import reactive, render, ui
from shinywidgets import render_widget

from app.services.augmentation_engine import AugmentationConfig, generate_augmented_dataset
from app.services.data_insights import detect_leakage_risk, profile_dataset, recommend_augmentation_preset, split_summary
from app.services.data_loader import load_ground_truth, load_scenario_metadata, load_timeseries, load_tunnel_config
from app.services.data_merger import build_merged_dataset
from app.services.dataset_builder import DatasetBuildConfig, add_training_targets, build_windowed_training_dataset, train_val_test_split
from app.services.evaluator import confusion_matrix_df, evaluate_predictions
from app.services.explainability import feature_importance_df
from app.services.export_service import export_augmented_files
from app.services.model_factory import ModelConfig, config_from_preset
from app.services.schema_validator import validate_schema
from app.services.trainer import TrainingConfig, available_devices, predict_with_model, train_model


def _file_path(fileinfo: list[dict[str, Any]] | None) -> str | None:
    return fileinfo[0]["datapath"] if fileinfo else None


def _model_layers(cfg: ModelConfig) -> pd.DataFrame:
    rows = [{"layer": 1, "type": "Input", "shape": f"({cfg.sequence_length},{cfg.input_dim})", "params": 0}]
    for idx in range(cfg.num_layers):
        units = cfg.hidden_dim if cfg.model_type in {"LSTM", "GRU", "1D CNN", "Hybrid CNN + LSTM"} else cfg.d_model
        rows.append({"layer": idx + 2, "type": cfg.model_type, "shape": f"({cfg.sequence_length},{units})", "params": int(units * (cfg.input_dim + 1))})
    rows.append({"layer": cfg.num_layers + 2, "type": "Output Heads", "shape": "event/risk/tte", "params": 3 * max(cfg.hidden_dim, cfg.d_model)})
    return pd.DataFrame(rows)




def _empty_figure(title: str, message: str = "Keine Daten"):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
    )
    return fig


def _sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def register_handlers(input, output, session) -> None:
    state = reactive.value(
        {
            "frames": None,
            "validation": None,
            "quality_profile": None,
            "augmented": None,
            "merged": None,
            "windowed": None,
            "splits": None,
            "split_summary": None,
            "leakage_findings": [],
            "model_config": None,
            "trained_model": None,
            "history": None,
            "predictions": None,
            "summary": None,
            "preset_hint": None,
            "training_cfg": None,
        }
    )

    def _get() -> dict[str, Any]:
        return state.get()

    def _set(**kwargs) -> None:
        cur = _get().copy()
        cur.update(kwargs)
        state.set(cur)

    @reactive.effect
    @reactive.event(input.split_701515)
    def _split701515():
        ui.update_slider("train_ratio", value=0.70)
        ui.update_slider("val_ratio", value=0.15)

    @reactive.effect
    @reactive.event(input.split_801010)
    def _split801010():
        ui.update_slider("train_ratio", value=0.80)
        ui.update_slider("val_ratio", value=0.10)

    @reactive.effect
    @reactive.event(input.split_602020)
    def _split602020():
        ui.update_slider("train_ratio", value=0.60)
        ui.update_slider("val_ratio", value=0.20)

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

        try:
            frames = {
                "tunnel_config": load_tunnel_config(paths["tunnel"]),
                "scenario_metadata": load_scenario_metadata(paths["meta"]),
                "timeseries": load_timeseries(paths["ts"]),
                "ground_truth": load_ground_truth(paths["gt"]),
            }
        except Exception as exc:
            ui.notification_show(f"Importfehler: {exc}", type="error")
            return

        _set(frames=frames, quality_profile=profile_dataset(frames))
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
    @reactive.event(input.suggest_preset)
    def _suggest_preset():
        s = _get()
        if not s["frames"]:
            ui.notification_show("Zuerst Daten laden.", type="warning")
            return
        hint = recommend_augmentation_preset(s["frames"]["scenario_metadata"], s["frames"]["ground_truth"])
        ui.update_select("aug_preset", selected=hint.preset)
        _set(preset_hint=hint)

    @reactive.effect
    @reactive.event(input.aug_preset)
    def _augmentation_preset_sync():
        p = input.aug_preset()
        presets = {
            "normal traffic": 0.25,
            "congestion": 0.45,
            "accident": 0.6,
            "fire": 0.7,
            "sensor fault": 0.5,
            "winter weather": 0.55,
            "heavy rain": 0.5,
            "mixed disturbance": 0.75,
        }
        if p in presets:
            ui.update_slider("aug_strength", value=presets[p])

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

        train_ratio = float(input.train_ratio())
        val_ratio = float(input.val_ratio())
        test_ratio = round(1.0 - train_ratio - val_ratio, 4)
        if test_ratio <= 0:
            ui.notification_show("Split ungültig: Train + Val muss < 1.0 sein.", type="error")
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
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        try:
            merged = build_merged_dataset(source)
            windowed = build_windowed_training_dataset(merged, cfg)
            windowed = add_training_targets(windowed, input.label_mode())
            splits = train_val_test_split(windowed, cfg)
        except Exception as exc:
            ui.notification_show(f"Dataset-Erstellung fehlgeschlagen: {exc}", type="error")
            return

        leakage = detect_leakage_risk(windowed, splits)
        split_info = split_summary(splits)
        _set(merged=merged, windowed=windowed, splits=splits, split_summary=split_info, leakage_findings=leakage)
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
            momentum=float(input.momentum()),
            device=input.device(),
            use_cuda=input.device() == "cuda",
            random_seed=int(input.train_seed()),
            class_weights=class_weights,
        )
        label_mode = "multi_task" if "target_event" in s["windowed"].columns else input.label_mode()
        train_df = s["splits"]["train"] if s["splits"] else s["windowed"]
        val_df = s["splits"]["val"] if s["splits"] else s["windowed"]

        try:
            model, history, summary = train_model(train_df, val_df, s["model_config"], t_cfg, label_mode)
        except Exception as exc:
            ui.notification_show(f"Training fehlgeschlagen: {exc}", type="error")
            return

        _set(trained_model=model, history=history, summary=summary, training_cfg=t_cfg)
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
            features = w[w["scenario_id"].astype(str) == sid].head(20)
        elif mode == "single window":
            idx = int(max(0, min(input.window_index(), len(w) - 1)))
            features = w.iloc[[idx]]
        else:
            size = int(max(1, min(input.batch_size_pred(), len(w))))
            features = w.head(size)

        pred = predict_with_model(s["trained_model"], features, top_n=int(input.top_n()))
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

        meta = s["frames"]
        if s["augmented"] is not None:
            meta = {
                "tunnel_config": s["frames"]["tunnel_config"],
                "scenario_metadata": s["augmented"]["augmented_scenario_metadata"],
                "timeseries": s["augmented"]["augmented_timeseries"],
                "ground_truth": s["augmented"]["augmented_ground_truth"],
            }

        files = export_augmented_files(
            input.export_dir(),
            meta["scenario_metadata"],
            meta["timeseries"],
            meta["ground_truth"],
            augmented_tunnel_config=meta["tunnel_config"],
            merged_training_dataset=s["merged"],
            windowed_dataset=s["windowed"],
            training_history=s["history"],
            training_summary=s["summary"],
            model_config_json=s["model_config"].to_json() if s["model_config"] else None,
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
    @render.text
    def data_quality_status():
        s = _get()
        if not s["quality_profile"]:
            return "Noch keine Datenanalyse verfügbar."
        parts = []
        if s["validation"] is not None:
            parts.append(f"Schema OK: {s['validation'].schema_ok}")
            if s["validation"].ml_findings:
                parts.append("ML-Findings: " + " | ".join(s["validation"].ml_findings[:5]))
        outliers = s["quality_profile"].get("outliers", {})
        if outliers:
            top = sorted(outliers.items(), key=lambda x: x[1], reverse=True)[:5]
            parts.append("Top Ausreißer: " + ", ".join([f"{k}:{v}" for k, v in top]))
        return "\n".join(parts)

    @output
    @render.data_frame
    def table_preview():
        frames = _get()["frames"]
        if not frames:
            return render.DataGrid(pd.DataFrame({"info": ["Noch keine Daten geladen"]}))
        sample = frames["scenario_metadata"].head(50)
        return render.DataGrid(sample)

    @output
    @render_widget
    def class_distribution_plot():
        qp = _get()["quality_profile"]
        if not qp or not qp.get("event_distribution"):
            return px.bar(pd.DataFrame({"klasse": ["n/a"], "anteil": [0]}), x="klasse", y="anteil", title="Klassenverteilung")
        df = pd.DataFrame({"klasse": list(qp["event_distribution"].keys()), "anteil": list(qp["event_distribution"].values())})
        df["anteil"] = pd.to_numeric(df["anteil"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return px.bar(df, x="klasse", y="anteil", title="Klassenverteilung")

    @output
    @render_widget
    def missing_plot():
        qp = _get()["quality_profile"]
        if not qp or not qp.get("missing_ratio"):
            return px.bar(pd.DataFrame({"feature": ["n/a"], "missing": [0]}), x="feature", y="missing", title="Missing Values")
        df = pd.DataFrame({"feature": list(qp["missing_ratio"].keys()), "missing": list(qp["missing_ratio"].values())})
        df["missing"] = pd.to_numeric(df["missing"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return px.bar(df, x="feature", y="missing", title="Missing Values (Top 15)")

    @output
    @render_widget
    def correlation_plot():
        qp = _get()["quality_profile"]
        corr = qp.get("correlation") if qp else None
        if corr is None or corr.empty:
            return _empty_figure("Korrelationen", "Noch keine Korrelationen verfügbar")
        cols = corr.columns[:20]
        corr_view = _sanitize_numeric(corr.loc[cols, cols])
        return px.imshow(corr_view, text_auto=False, title="Feature-Korrelationen")

    @output
    @render_widget
    def feature_distribution_plot():
        frames = _get()["frames"]
        if not frames:
            return px.bar(pd.DataFrame({"feature": ["no_data"], "count": [0]}), x="feature", y="count", title="Feature-Verteilung")
        ts = frames["timeseries"]
        col = "speed_mean_kmh" if "speed_mean_kmh" in ts.columns else (ts.select_dtypes(include="number").columns[0] if not ts.select_dtypes(include="number").empty else None)
        if col is None:
            return px.bar(pd.DataFrame({"feature": ["none"], "count": [0]}), x="feature", y="count", title="Keine numerischen Features")
        return px.histogram(ts, x=col, nbins=40, title=f"Verteilung: {col}")

    @output
    @render_widget
    def event_timeline_plot():
        frames = _get()["frames"]
        if not frames:
            return _empty_figure("Event-Timeline", "Noch keine Event-Daten")
        gt = frames["ground_truth"].copy()
        if not {"timestamp_s", "label_event_active"}.issubset(gt.columns):
            return _empty_figure("Event-Timeline", "Noch keine Event-Daten")
        gt["label_event_active"] = pd.to_numeric(gt["label_event_active"], errors="coerce").fillna(0)
        timeline = gt.groupby("timestamp_s", as_index=False)["label_event_active"].mean()
        return px.line(timeline, x="timestamp_s", y="label_event_active", title="Event-Verlauf (Aktivitätsrate)")

    @output
    @render.text
    def preset_reason():
        hint = _get()["preset_hint"]
        if hint is None:
            return "Noch kein Preset-Vorschlag berechnet."
        return f"Vorschlag: {hint.preset} | Confidence: {hint.confidence:.2f} | Grund: {hint.reason}"

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
        extra = ""
        if s["split_summary"]:
            extra = "\n" + json.dumps(s["split_summary"], indent=2, ensure_ascii=False)
        if s["leakage_findings"]:
            extra += "\nWARNUNGEN: " + " | ".join(s["leakage_findings"])
        return f"Windowed Samples: {len(s['windowed'])}\nSplits: {split_sizes}{extra}"

    @output
    @render_widget
    def split_distribution_plot():
        s = _get()
        if not s["splits"]:
            return _empty_figure("Split-Verteilung", "Noch keine Split-Daten")
        rows = []
        for split_name, df in s["splits"].items():
            rows.append({"split": split_name, "samples": len(df), "scenarios": df["scenario_id"].nunique() if "scenario_id" in df.columns else 0})
        plot_df = pd.DataFrame(rows)
        plot_df[["samples", "scenarios"]] = plot_df[["samples", "scenarios"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return px.bar(plot_df, x="split", y="samples", color="scenarios", title="Samples pro Split")

    @output
    @render_widget
    def feature_plot():
        windowed = _get()["windowed"]
        if windowed is None or windowed.empty:
            return _empty_figure("Features", "Noch keine Features verfügbar")
        numeric = windowed.select_dtypes(include="number")
        if numeric.empty:
            return _empty_figure("Features", "Keine numerischen Features")
        means = numeric.mean().replace([np.inf, -np.inf], np.nan).fillna(0).sort_values(ascending=False).head(15)
        df = pd.DataFrame({"feature": means.index.astype(str), "value": pd.to_numeric(means.values, errors="coerce")}).fillna(0)
        return px.bar(df, x="feature", y="value", title="Top Feature-Mittelwerte")

    @output
    @render.data_frame
    def windowed_table():
        w = _get()["windowed"]
        return render.DataGrid(w.head(100) if isinstance(w, pd.DataFrame) else pd.DataFrame())

    @output
    @render.text
    def device_status():
        devices = available_devices()
        return f"CPU verfügbar: {devices['cpu']} | CUDA verfügbar: {devices['cuda']}"

    @output
    @render.text
    def training_config_summary():
        s = _get()
        cfg = s.get("training_cfg")
        if cfg is None:
            return "Noch kein Training gestartet."
        return json.dumps(cfg.__dict__, indent=2, ensure_ascii=False)

    @output
    @render.text
    def training_status():
        s = _get()
        if not s["summary"]:
            return "Training wurde noch nicht ausgeführt."
        return json.dumps(s["summary"], indent=2, ensure_ascii=False)

    @output
    @render_widget
    def model_architecture_plot():
        s = _get()
        cfg = s["model_config"]
        if cfg is None:
            return _empty_figure("Modellarchitektur", "Keine Modellkonfiguration gespeichert")
        layers = _model_layers(cfg)
        return px.bar(layers, x="layer", y="params", color="type", hover_data=["shape"], title="Modellarchitektur (Layer/Parameter)")

    @output
    @render_widget
    def training_plot():
        history = _get()["history"]
        if history is None or history.empty:
            return _empty_figure("Training Metriken", "Noch keine Trainingshistorie")
        hist = history.copy()
        metric_cols = [c for c in ["train_loss", "val_loss", "accuracy", "precision", "recall", "f1"] if c in hist.columns]
        for c in metric_cols:
            hist[c] = pd.to_numeric(hist[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return px.line(hist, x="epoch", y=metric_cols, title="Training Metriken")

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
    def prediction_timeline_plot():
        preds = _get()["predictions"]
        if preds is None or preds.empty:
            return _empty_figure("Prediction Verlauf", "Noch keine Predictions")
        if "window_end_s" not in preds.columns:
            preds = preds.reset_index().rename(columns={"index": "window_end_s"})
        pred_view = preds.sort_values("window_end_s").copy()
        pred_view["confidence"] = pd.to_numeric(pred_view.get("confidence", 0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pred_view["predicted_event_type"] = pred_view.get("predicted_event_type", "unknown").astype(str)
        return px.line(pred_view, x="window_end_s", y="confidence", color="predicted_event_type", title="Vorhersage-Verlauf")

    @output
    @render_widget
    def importance_plot():
        model = _get()["trained_model"]
        if model is None:
            return _empty_figure("Feature Importance", "Noch kein Modell trainiert")
        fi = feature_importance_df(model).head(20).copy()
        if fi.empty:
            return px.bar(pd.DataFrame({"feature": ["none"], "importance": [0.0]}), x="feature", y="importance", title="Feature Importance")
        fi["importance"] = pd.to_numeric(fi["importance"], errors="coerce").replace([float("inf"), float("-inf")], 0).fillna(0)
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
