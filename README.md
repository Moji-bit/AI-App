# Tunnel Time-Series AI App

## Ziel
Diese App unterstützt die vollständige Pipeline:

**Data -> Architecture -> Training -> Inspect -> Predict -> Analysis**

1. Rohdaten importieren
2. Dataset validieren
3. Data Augmentation ausführen
4. großes Trainingsdataset erzeugen
5. KI-Modell konfigurieren
6. Modell trainieren
7. Modell evaluieren
8. Vorhersagen + Szenarioanalyse

## Tab-Struktur
1. Introduction
2. Data Import
3. Dataset Validation
4. Data Augmentation
5. Training Dataset Builder
6. Model Architecture
7. Training
8. Inspect / Explainability
9. Predict / Inference
10. Analysis / Evaluation

## 4-Dateien-Schema
- `tunnel_config.csv`
- `scenario_metadata.csv`
- `timeseries.csv`
- `ground_truth.csv`

## Architektur
- `app/services/data_loader.py`
- `app/services/schema_validator.py`
- `app/services/data_merger.py`
- `app/services/augmentation_engine.py`
- `app/services/dataset_builder.py`
- `app/services/model_factory.py`
- `app/services/trainer.py`
- `app/services/evaluator.py`
- `app/services/explainability.py`
- `app/services/export_service.py`
- `app/ui/app.py`
- `app/ui/pages/*.py`

## Start
```bash
pip install -r requirements.txt
python run_dataset_builder.py
```

oder

```bash
streamlit run app/ui/app.py
```

## Beispiel-Workflow
1. **Data Import**: 4 CSV-Dateien hochladen und Vorschau prüfen.
2. **Dataset Validation**: Fehler/Warnungen/Passed Checks prüfen.
3. **Data Augmentation**: Zielanzahl, Stärke, Noise, Shift, Missing, Wetter, Klassenbalance setzen und augmentieren.
4. **Training Dataset Builder**: Merge + Sliding Windows + Label-Modus + Train/Val/Test Split erzeugen.
5. **Model Architecture**: Preset wählen oder Parameter manuell setzen; JSON export/import.
6. **Training**: Trainingsparameter setzen und Lauf starten.
7. **Inspect / Explainability**: Feature Importance, Attention-Proxy, FP/FN Analyse.
8. **Predict / Inference**: Einzel-/Batch-Vorhersagen.
9. **Analysis / Evaluation**: Confusion Matrix + klassische und operationale Metriken.

## Hinweise
- Augmentation enthält ereignislogische Regeln (accident/congestion/fire/sensor_fault).
- Quality Score pro augmentiertem Szenario wird in den Metadaten gespeichert.
- Export umfasst augmentierte CSVs, merged Training Dataset und optional windowed Dataset.
