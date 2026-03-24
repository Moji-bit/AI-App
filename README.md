# Tunnel AI App – Shiny for Python Migration

Diese Anwendung wurde vollständig auf **Shiny for Python** migriert und als moderne, workflow-orientierte Analytics-App neu aufgebaut.

## Start

```bash
pip install -r requirements.txt
shiny run --reload app.py
```

Alternativ:

```bash
python main.py
```

## Neue Architektur

```text
app.py                               # zentraler Shiny-Einstiegspunkt
app/
  services/                          # Fachlogik (Import, Validierung, Augmentation, Training, Evaluation)
  shiny_app/
    application.py                   # App-Wiring (UI + Server)
    ui/
      layout.py                      # komplette Seiten-/Workflow-Komposition
      theme.py                       # modernes CSS-Theme
    server/
      handlers.py                    # reaktive Handler, State, Event-Workflows
tests/
  test_dataset_builder.py            # Kern-Tests für Pipeline-Funktionen
main.py                              # lokaler Start-Wrapper
requirements.txt
```

## Funktionsumfang (migriert)

- Upload der 4 Rohdaten-Dateien (`tunnel_config`, `scenario_metadata`, `timeseries`, `ground_truth`)
- Schema- und Konsistenzvalidierung
- Datenaugmentation mit parametrischer Konfiguration
- Building von Windowed-Trainingsdaten inkl. Split (train/val/test)
- Modellkonfiguration + Training
- Inferenz (single scenario/window/batch)
- Evaluation (Metriken + Confusion Matrix)
- Explainability (Feature Importance)
- Export der erzeugten Artefakte

## UI/UX-Konzept

Die Shiny-Oberfläche folgt einem klaren Produkt-Workflow:

1. **Data Intake**: Upload + Validierung
2. **Augmentation**: Generierung zusätzlicher Szenarien
3. **Dataset Builder**: Feature-/Window-Building und Splits
4. **Model & Training**: Modell- und Trainingsparameter, Trainingsergebnisse
5. **Inference & Evaluation**: Vorhersage, Metriken, Matrix, Importance
6. **Export**: Persistenz der erzeugten Datensätze

Die App verwendet:

- modernes Navbar-Layout
- Hero-Bereich + KPI-Kacheln
- card-basiertes Section-Design
- konsistente Abstände und Typografie
- klare Trennung von Eingabe, Verarbeitung und Ergebnisdarstellung

## Breaking Changes

- Die alte Streamlit-Oberfläche unter `app/ui/*` wurde entfernt.
- Startbefehl ist jetzt `shiny run --reload app.py` statt `streamlit run ...`.
