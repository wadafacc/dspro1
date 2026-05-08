# Predicting Apartment Rental Prices in Switzerland

> HSLU DSPRO1 — Team 8 — Machine-Learning-Modell zur Vorhersage von Kaltmieten Schweizer Wohnungen aus Wohnungs-, Geo- und Lagedaten.

---

## Was macht das Projekt?

Wir trainieren mehrere Regressions-Modelle (Linear Regression, Random Forest, XGBoost, LightGBM, Stacking) auf einem zusammengetragenen Datensatz Schweizer Mietwohnungen, vergleichen sie systematisch und bauen daraus eine produktionsreife Pipeline (`RentPredictor`), die aus den Eingabe-Features eine Mietpreis-Vorhersage liefert — inklusive Konfidenzintervall, Stabilitäts-Check und Modell-Karte.

Demo: `streamlit run src/app.py` öffnet ein interaktives Frontend, in dem du Wohnungs-Parameter eingeben und sofort eine Preis-Schätzung bekommen kannst.

## Quick Start

```bash
# 1. Repository klonen
git clone <repo-url>
cd dspro1

# 2. (Empfohlen) Virtuelles Environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. Notebook öffnen (volle Pipeline + Analyse, 24 Kapitel)
jupyter lab src/notebooks/model_v3_clean.ipynb

# 5. Demo-App starten (interaktive Vorhersage im Browser)
streamlit run src/app.py
```

## Projektstruktur

```
dspro1/
├── README.md                       <- du bist hier
├── requirements.txt                <- Python-Abhängigkeiten
├── .gitignore
├── docs/                           <- Berichte, Präsentationen, AI Canvas
│   ├── ai-canvas/
│   ├── final-report/
│   ├── presentation-mid-term/
│   ├── project-proposal/
│   └── schemes/                    <- drawio Architektur-Diagramme
├── src/
│   ├── app.py                      <- Streamlit Demo-App
│   ├── notebooks/
│   │   ├── model_v3_clean.ipynb    <- Hauptnotebook (24 Kapitel)
│   │   ├── model_v2.ipynb          <- Vorgänger, als Referenz
│   │   ├── archive/, py-tests/, catboost_info/
│   ├── external-sources/           <- Daten-Sync-Notebooks (GWR, swisstopo)
│   │   └── output_csv/model.csv    <- Trainings-/Eval-Daten
│   ├── scrapegoat/                 <- Rust-basierter Scraper
│   ├── rentables-scraper/
│   └── .env.example
└── models/                         <- gespeicherte Modell-Artefakte (.joblib)
                                       wird beim ersten Notebook-Lauf erstellt
```

## Was steckt im Hauptnotebook (`model_v3_clean.ipynb`)?

24 Kapitel, ca. 186 Zellen. In Kürze:

| Kapitel | Inhalt |
|---|---|
| 1-7 | Setup, Datenladen, Spalten-Rename, Datenqualität, Outlier-Filter, Feature Engineering, zentrale Feature-Sets |
| 8 | Sauberer Train/Eval-Split (kein `concat → head/tail` Recovery wie in v2) |
| 9-13 | Modell-Pipelines, Vergleich, Train-vs-Eval-Metriken, Overfitting-Diagnose, Cross-Validation |
| 14-16 | Visualisierungen: Actual-vs-Predicted, Prediction Curves, Polynomial-Demo, Residuen, Fehleranalyse nach Preisgruppen |
| 17-19 | Feature Importance, Modellauswahl, Final Prediction mit Spalten-Check |
| 20-21 | Optionaler Group-Split, Geo-Analyse + KMeans/DBSCAN-Clustering, KNN-Distance-Features |
| 22 | Erweiterte Verbesserungen: Imputation, Log-Target, RandomizedSearchCV-Tuning, Stratified Split, Learning/Validation Curves, Quantile Regression, SHAP, Outlier-Detection (IsolationForest), RFECV, joblib-Persistenz, Drift-Check, Bias-Analyse |
| 23 | Runde 2: Multi-Metric-Vergleich, Stacking, Q-Q & Heteroskedastizität, PDP, Bootstrap-CIs, HalvingRandomSearch, Conformal Prediction (MAPIE), KNN-Distance-Features, statistischer Modellvergleich, Modell-Karte, requirements.txt-Generator |
| 24 | Runde 3: End-to-End `RentPredictor`-Klasse, Sanity-Tests, v2 vs v3 Vergleich, Datasheet, Hold-Out Train/Val/Test, vollständige Artefakt-Persistenz |

## Datenquellen

- **GWR** (Gebäude- und Wohnungsregister) — siehe `src/external-sources/gwr_egid_db_sync.ipynb`
- **swisstopo** (Geo-Koordinaten LV95, Höhe) — siehe `src/external-sources/swisstopo_enrich_db_sync_v2.ipynb`
- Eigener Scraper für Vermietungsplattformen: `src/scrapegoat/` (Rust) und `src/rentables-scraper/`

Aufbereitetes Trainings-Set: `src/external-sources/output_csv/model.csv` (~4'500 Wohnungen × 12 Spalten).

## Reproduzierbarkeit

- `RANDOM_STATE = 42` durchgängig
- `requirements.txt` mit Versions-Pins
- `models/rent_predictor_v3.joblib` enthält die finale Pipeline + Metadata (training_date, python_version, test_metrics)
- Modell-Karte und Datasheet sind im Notebook (Kap. 23.15 / 24.5) dokumentiert

## Team und Lizenz

- **Team 8 — DSPRO1 HSLU** (Hochschule Luzern)
- Maintainer: Elias Martinelli
- Status: in Bearbeitung
- Lizenz: tbd (akademisches Projekt)
