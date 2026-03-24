# Workflow Testplan (Import → Validate → Augment → Build → Train → Predict → Export)

Dieser Testplan ist für manuelle End-to-End-Prüfungen der Shiny-App gedacht.

## 0) Start (Windows/Conda)

```bash
conda activate tunnel-ai-app
pip install -r requirements.txt
shiny run --reload app.py
```

App im Browser öffnen: `http://127.0.0.1:8000`

---

## 1) Data Intake

1. Upload der vier CSV-Dateien:
   - `tunnel_config.csv`
   - `scenario_metadata.csv`
   - `timeseries.csv`
   - `ground_truth.csv`
2. `Dateien laden` klicken.
3. `Schema validieren` klicken.

### Erwartung
- Notification „Daten erfolgreich geladen“.
- `validation_summary` zeigt `schema_ok` und ggf. `warnings`, `ml_findings`, `duplicate_rows`.
- Keine Python-Exception im Terminal.

---

## 2) Data Analysis

1. Tab **Data Analysis** öffnen.
2. Prüfen:
   - Klassenverteilung
   - Missing Values
   - Korrelationsmatrix
   - Feature-Verteilung
   - Event-Timeline

### Erwartung
- Alle Plots rendern ohne Fehler.
- Leere Zustände werden als Fallback-Plot gezeigt (kein Crash).

---

## 3) Augmentation

1. Optional `Preset automatisch erkennen` klicken.
2. `Augmentation starten` klicken.

### Erwartung
- `augmentation_status` zeigt Anzahl augmentierter Szenarien.
- Keine Fehler wie `assignment destination is read-only`.
- Tabelle `augmented_meta_table` wird gefüllt.

---

## 4) Dataset Builder

1. Tab **Dataset Builder** öffnen.
2. Quick-Preset klicken (`70/15/15`, `80/10/10` oder `60/20/20`).
3. `Training Dataset bauen` klicken.

### Erwartung
- `dataset_status` zeigt Sample- und Split-Infos.
- `split_distribution_plot` und `feature_plot` rendern.
- Keine Shuffle-Warnung mit ArrowStringArray.

---

## 5) Model & Training

1. Modell-Preset auswählen.
2. Optional Device `cpu`/`cuda` wählen.
3. `Konfiguration speichern` klicken.
4. `Training starten` klicken.

### Erwartung
- `training_config_summary` zeigt aktive Parameter.
- `training_plot` zeigt train/val loss + accuracy/precision/recall/F1.
- Kein Fehler `unsupported type: <class 'float'>`.

---

## 6) Inference & Evaluation

1. Modus wählen (`single scenario`, `single window`, `batch`).
2. `Inference starten` klicken.

### Erwartung
- `prediction_table` enthält `predicted_event_type`, `confidence`, `top_classes`.
- Confusion Matrix / Timeline rendern ohne JSON-Serialisierungsfehler.

---

## 7) Export

1. Tab **Export** öffnen.
2. Export-Verzeichnis setzen.
3. `Artefakte exportieren` klicken.

### Erwartung
- `export_status` enthält Pfade mit Timestamp-Dateinamen.
- Mindestens vorhanden: augmentierte CSVs; optional windowed/history/summary/model_config.

---

## 8) Regression-Checks im Terminal

```bash
pytest -q
python -m compileall app tests
```

### Erwartung
- Alle Tests grün.
- Keine Compile-Fehler.
