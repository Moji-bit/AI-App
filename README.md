# Tunnel AI App – Dataset Builder & Augmentation Pipeline

Diese Implementierung ergänzt die App um einen vollständigen **Dataset Builder** für das 4-Dateien-Schema:

1. `tunnel_config.csv`
2. `scenario_metadata.csv`
3. `timeseries.csv`
4. `ground_truth.csv`

## Architektur

- `app/services/data_loader.py`
  - Lädt und normalisiert CSV-Dateien (`load_tunnel_config`, `load_scenario_metadata`, `load_timeseries`, `load_ground_truth`).
- `app/services/schema_validator.py`
  - Validiert Pflichtspalten, numerische Datentypen, fehlende Werte, Plausibilitäten und Cross-File-Konsistenz.
- `app/services/data_merger.py`
  - Baut ein konsistentes Trainingsdataset über Joins auf `scenario_id`, `timestamp_s`, `tunnel_id`.
- `app/services/augmentation_engine.py`
  - Kern-Augmentation (Signal-Augmentation, event-aware Regeln, Klassen-Balance, Quality Score).
- `app/services/scenario_generator.py`
  - Presets, Szenario-Summary und optionale Window-Sequenzen für LSTM/Transformer.
- `app/services/export_service.py`
  - Exportiert augmentierte CSV-Dateien und optional merged/windowed Datasets.
- `app/ui/pages/dataset_builder.py`
  - Streamlit-UI für Upload, Report, Konfiguration, Preview und Export.

## Starten

```bash
pip install -r requirements.txt
python run_dataset_builder.py
```

oder direkt:

```bash
streamlit run app/ui/pages/dataset_builder.py
```

## Beispiel-Workflow

1. Alle vier CSV-Dateien hochladen.
2. Schema-Report prüfen (Fehler/Warnungen).
3. Szenario-Zusammenfassung ansehen (Events, Wetter, Severity, Dauer).
4. Augmentation konfigurieren:
   - Ziel-Szenarien
   - Stärke/Noise
   - Event-Shift, Missing/Outlier-Rate
   - Klassen-Balance Ziele
   - Wettervariationen/Presets
5. `Generate Augmented Dataset` klicken.
6. Preview Original vs. Augmentiert über Metriken (`speed`, `flow`, `occupancy`, `co`, `visibility`, `queue`).
7. Optional Windowed Sequences erzeugen.
8. Exportieren als:
   - `augmented_timeseries.csv`
   - `augmented_ground_truth.csv`
   - `augmented_scenario_metadata.csv`
   - optional `augmented_tunnel_config.csv`
   - optional `merged_training_dataset.csv`
   - optional `windowed_sequences.csv`

## Hinweise zur Reproduzierbarkeit

- Jeder Lauf kann über `Random Seed` reproduzierbar gemacht werden.
- Jede augmentierte Instanz bekommt eine neue `scenario_id`.
- Ground-Truth wird nach Event-Shift und Event-Dauer automatisch synchronisiert.
