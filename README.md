# Tunnel Time-Series AI App

Eine tab-basierte Streamlit-App für eine vollständige Tunnel-AI-Pipeline:

**Data Import -> Validation -> Augmentation -> Dataset Builder -> Model -> Training -> Inspect -> Predict -> Analysis**

## Aktuelle Features

### 1) Data Import
- Upload von `tunnel_config.csv`, `scenario_metadata.csv`, `timeseries.csv`, `ground_truth.csv`
- Datei-Status inkl. Größe
- Tabellen-Preview + Spaltenübersicht je Datei

### 2) Dataset Validation
- Pflichtspaltenprüfung
- Datentypprüfung
- `scenario_id` / `tunnel_id` Cross-File-Konsistenz
- `timestamp_s` Synchronität (`timeseries` vs `ground_truth`)
- Missing Values
- Duplikatprüfung (`scenario_id`, `(scenario_id,timestamp_s)`)
- Plausibilitätschecks (u.a. occupancy/visibility/fan power)

### 3) Data Augmentation
- Numerische Augmentation: Noise, Drift, Skalierung, Lag, Smoothing, Outlier, Missing, Sampling-Variation
- Szenario-Augmentation: Event-Start/Dauer/Ort, Wetter, Entry-Flow, Heavy-Vehicle-Anteil, Temp/Wind
- Event-aware Regeln: `accident`, `congestion`, `fire`, `sensor_fault`
- Class-Balancing über Zielverteilung
- Neue `scenario_id` je augmentiertem Szenario
- Quality Score + Mindestqualität als Filter

### 4) Training Dataset Builder
- Merge der 4 Datenquellen
- Sliding Window Builder (`sequence_length`, `forecast_horizon`, `stride`)
- Label-Modi:
  - event classification
  - risk classification
  - time-to-event regression
  - multi-task
- Train/Val/Test Split auf Szenarioebene
- Export von `merged_training_dataset.csv` + optional windowed dataset

### 5) Model Architecture
- Presets:
  - `baseline_lstm`
  - `baseline_gru`
  - `baseline_1dcnn`
  - `transformer_small`
  - `transformer_multitask`
  - `hybrid_cnn_lstm`
  - `hybrid_cnn_transformer`
- Hyperparameter-Editor + JSON Import/Export

### 6) Training
- Konfigurierbar: epochs, batch_size, learning_rate, optimizer, scheduler, early stopping, patience, seed, class weights
- Reale Modellanpassung über `scikit-learn` (statt reinem Dummy)
- Training History + best model summary

### 7) Inspect / Explainability
- Feature-Importance
- Attention-Proxy
- FP/FN Breakdown

### 8) Predict / Inference
- Single scenario, single window, batch prediction
- Outputs: predicted event, risk level, class probabilities, risk score, event location, time-to-event, uncertainty

### 9) Analysis / Evaluation
- Confusion Matrix
- Accuracy / Precision / Recall / F1
- ROC-AUC / PR-AUC (binär event vs normal, falls möglich)
- Lead-Time, False Alarm Rate, Missed Event Rate, Robustheit unter Sensorfehlern

---

## Projektstruktur

```text
app/
  services/
    data_loader.py
    schema_validator.py
    data_merger.py
    augmentation_engine.py
    dataset_builder.py
    model_factory.py
    trainer.py
    evaluator.py
    explainability.py
    export_service.py
    scenario_generator.py
  ui/
    app.py
    pages/
      introduction.py
      data_import.py
      dataset_validation.py
      data_augmentation.py
      training_dataset_builder.py
      model_architecture.py
      training.py
      inspect_explainability.py
      predict_inference.py
      analysis_evaluation.py
tests/
run_dataset_builder.py
requirements.txt
environment.yml
```

---

## Setup

### Option A: pip
```bash
pip install -r requirements.txt
python run_dataset_builder.py
```

### Option B: conda (Windows-freundlich)
```bash
conda env create -f environment.yml
conda activate tunnel-ai-app
python run_dataset_builder.py
```

Alternativ direkt:
```bash
streamlit run app/ui/app.py
```

---

## CSV-Schema (Kurz)

1. `tunnel_config.csv` – eine Zeile pro Tunnelkonfiguration
2. `scenario_metadata.csv` – eine Zeile pro Szenario
3. `timeseries.csv` – viele Zeilen pro Szenario (Zeitpunkte)
4. `ground_truth.csv` – Labels pro Zeitpunkt

Die App validiert diese Struktur und merged über `scenario_id`, `timestamp_s` und `tunnel_id`.

---

## Empfohlener Workflow

1. **Data Import**: 4 CSVs laden
2. **Dataset Validation**: Errors/Warnungen beheben
3. **Data Augmentation**: Zielanzahl + Qualität + Klassenbalance einstellen
4. **Training Dataset Builder**: windows + targets + split erzeugen
5. **Model Architecture**: Preset/JSON wählen
6. **Training**: Modell trainieren
7. **Predict / Inference**: Vorhersagen erzeugen
8. **Analysis / Evaluation**: Metriken + Confusion Matrix prüfen

---

## Tests

```bash
pytest -q
```

Enthaltene Basistests:
- schema validation
- augmentation output consistency
- dataset merging
- training dataset build pipeline
- model config JSON roundtrip
