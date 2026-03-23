from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("Introduction")
    st.markdown(
        """
        Diese App bildet den End-to-End-Workflow für Tunnel-/Zeitreihendaten ab:

        **Data -> Architecture -> Training -> Inspect -> Predict -> Analysis**

        1. Rohdaten importieren
        2. Dataset validieren
        3. Data Augmentation ausführen
        4. Trainingsdataset bauen
        5. Modell konfigurieren
        6. Modell trainieren
        7. Vorhersagen und Analyse durchführen
        """
    )
