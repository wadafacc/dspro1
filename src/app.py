"""Streamlit Demo-App für die Mietpreis-Vorhersage.

Start:
    streamlit run src/app.py

Beim ersten Start wird das Modell aus der CSV-Datei trainiert und gecached.
Folgende Aufrufe lesen das gecachte Modell direkt — das macht die UI schnell.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ===========================================================================
# Pfade
# ===========================================================================
APP_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_PATH  = APP_DIR / "external-sources" / "output_csv" / "model.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "rent_predictor_streamlit.joblib"


# ===========================================================================
# RentPredictor (inline, identisch zur Notebook-Klasse Kapitel 24.2)
# ===========================================================================
class RentPredictor:
    """End-to-End-Pipeline fuer die Mietpreis-Vorhersage."""

    def __init__(
        self,
        target_col: str = "price",
        feature_cols=None,
        n_geo_clusters: int = 8,
        knn_k: int = 10,
        outlier_min_area: int = 10,
        outlier_min_price: int = 300,
        reference_year: int = 2026,
        random_state: int = 42,
        model=None,
    ):
        self.target_col        = target_col
        self.feature_cols      = feature_cols
        self.n_geo_clusters    = n_geo_clusters
        self.knn_k             = knn_k
        self.outlier_min_area  = outlier_min_area
        self.outlier_min_price = outlier_min_price
        self.reference_year    = reference_year
        self.random_state      = random_state
        self.model             = model

    def _clean(self, df, training=True):
        out = df.copy()
        if training:
            mask = out["area"] >= self.outlier_min_area
            if self.target_col in out.columns:
                mask &= out[self.target_col] >= self.outlier_min_price
            out = out[mask].drop_duplicates().reset_index(drop=True)
        return out

    def _engineer(self, df):
        out = df.copy()
        if "year_built" in out.columns:
            out["building_age"] = self.reference_year - out["year_built"]
        if "rooms" in out.columns and "area" in out.columns:
            out["area_per_room"] = np.where(
                out["rooms"] > 0, out["area"] / out["rooms"], np.nan
            )
        if "apartments" in out.columns and "land_area" in out.columns:
            out["land_area_per_apartment"] = np.where(
                out["apartments"] > 0, out["land_area"] / out["apartments"], np.nan,
            )
        return out

    def _geo_cluster(self, df, fit=False):
        cols = ["east", "north"]
        if not all(c in df.columns for c in cols):
            return df
        if fit:
            self._geo_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("km", KMeans(n_clusters=self.n_geo_clusters,
                              random_state=self.random_state, n_init=10)),
            ])
            self._geo_pipe.fit(df[cols])
        out = df.copy()
        out["geo_cluster"] = self._geo_pipe.predict(out[cols])
        return out

    def _knn_features(self, df, fit=False):
        cols = ["east", "north"]
        if not all(c in df.columns for c in cols):
            return df
        if fit:
            self._coord_scaler = StandardScaler().fit(df[cols])
            self._train_coords = self._coord_scaler.transform(df[cols])
            self._train_prices = df[self.target_col].values
            self._nbrs = NearestNeighbors(n_neighbors=self.knn_k + 1).fit(
                self._train_coords
            )
            _, idx = self._nbrs.kneighbors(self._train_coords)
            idx = idx[:, 1:]
            out = df.copy()
            out["knn_price_mean"]   = self._train_prices[idx].mean(axis=1)
            out["knn_price_median"] = np.median(self._train_prices[idx], axis=1)
            return out
        coords_q = self._coord_scaler.transform(df[cols])
        _, idx = self._nbrs.kneighbors(coords_q, n_neighbors=self.knn_k)
        out = df.copy()
        out["knn_price_mean"]   = self._train_prices[idx].mean(axis=1)
        out["knn_price_median"] = np.median(self._train_prices[idx], axis=1)
        return out

    def fit(self, df):
        df_c = self._clean(df, training=True)
        df_e = self._engineer(df_c)
        df_g = self._geo_cluster(df_e, fit=True)
        df_k = self._knn_features(df_g, fit=True)

        if self.feature_cols is None:
            base = ["east", "north", "elevation", "area", "rooms", "year_built",
                    "apartments", "land_area", "population", "oev", "solar",
                    "building_age", "area_per_room", "land_area_per_apartment",
                    "geo_cluster", "knn_price_mean", "knn_price_median"]
            self.feature_cols = [c for c in base if c in df_k.columns]

        self._imputer = SimpleImputer(strategy="median")
        X = self._imputer.fit_transform(df_k[self.feature_cols])
        y = df_k[self.target_col].values

        if self.model is None:
            if HAS_LGBM:
                self.model = LGBMRegressor(
                    n_estimators=500, learning_rate=0.05,
                    random_state=self.random_state, n_jobs=-1, verbose=-1,
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=300, random_state=self.random_state,
                    n_jobs=-1, min_samples_leaf=2,
                )
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, df):
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("RentPredictor wurde noch nicht gefittet.")
        df_c = self._clean(df, training=False)
        df_e = self._engineer(df_c)
        df_g = self._geo_cluster(df_e, fit=False)
        df_k = self._knn_features(df_g, fit=False)
        for c in self.feature_cols:
            if c not in df_k.columns:
                df_k[c] = np.nan
        X = self._imputer.transform(df_k[self.feature_cols])
        return self.model.predict(X)


# ===========================================================================
# Daten- und Modell-Loader (mit Caching)
# ===========================================================================
COLUMN_RENAMES = {
    "area_sqm":    "area",
    "rooms":       "rooms",
    "price_cold":  "price",
    "population":  "population",
    "oev_score":   "oev",
    "solar_class": "solar",
    "elevation_m": "elevation",
    "lv95_east":   "east",
    "lv95_north":  "north",
    "gbauj":       "year_built",
    "ganzwhg":     "apartments",
    "garea":       "land_area",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Daten nicht gefunden: {DATA_PATH}. "
            "Stelle sicher, dass src/external-sources/output_csv/model.csv existiert."
        )
    df = pd.read_csv(DATA_PATH).rename(columns=COLUMN_RENAMES)
    return df


@st.cache_resource(show_spinner="Trainiere Modell (einmalig, ca. 10 Sekunden) ...")
def get_predictor() -> tuple[RentPredictor, str]:
    """Lädt gecachten Predictor oder trainiert frisch."""
    df = load_data()

    # Versuche, ein bereits trainiertes Streamlit-Artefakt zu laden
    if MODEL_PATH.exists():
        try:
            artifact = joblib.load(MODEL_PATH)
            return artifact["predictor"], f"Cache: {MODEL_PATH.name}"
        except Exception as exc:
            st.warning(f"Konnte gecachtes Modell nicht laden ({exc}), trainiere neu.")

    # Frisch trainieren
    rp = RentPredictor()
    rp.fit(df)

    # Optional: Cache schreiben
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"predictor": rp}, MODEL_PATH)
    except Exception:
        pass  # nicht kritisch

    return rp, "frisch trainiert"


# ===========================================================================
# UI
# ===========================================================================
st.set_page_config(
    page_title="Mietpreis-Schätzer Schweiz",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Mietpreis-Schätzer Schweiz")
st.caption("HSLU DSPRO1 Team 8 — Demo der `RentPredictor`-Pipeline aus `model_v3_clean.ipynb`")

# Modell laden
try:
    predictor, model_source = get_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.sidebar.success(f"Modell: {type(predictor.model).__name__} ({model_source})")
st.sidebar.divider()

# ---------- Wohnungs-Eigenschaften ----------
st.sidebar.header("🏢 Wohnung")
area       = st.sidebar.slider("Wohnfläche (m²)",      20, 250, 75)
rooms      = st.sidebar.slider("Zimmer",                 1,   8,  3)
year_built = st.sidebar.slider("Baujahr",             1900, 2026, 1990)

st.sidebar.header("🏗️ Gebäude")
apartments = st.sidebar.slider("Wohnungen im Gebäude",   1, 100,  8)
land_area  = st.sidebar.slider("Grundstücksfläche (m²)", 50, 2000, 300)

# ---------- Lage ----------
st.sidebar.header("📍 Lage")
LOCATION_PRESETS = {
    "Zürich (HB)":     (2683000, 1247000,  408),
    "Bern (HB)":       (2600000, 1200000,  540),
    "Luzern (HB)":     (2666000, 1211000,  435),
    "Genf (HB)":       (2500000, 1118000,  375),
    "Basel (SBB)":     (2611000, 1267000,  270),
    "Lugano (Centro)": (2717500, 1095500,  273),
    "Zermatt":         (2624500, 1097000, 1620),
    "Custom":          None,
}
preset = st.sidebar.selectbox("Stadt-Preset", list(LOCATION_PRESETS.keys()))
if LOCATION_PRESETS[preset] is not None:
    east_d, north_d, elev_d = LOCATION_PRESETS[preset]
else:
    east_d, north_d, elev_d = 2683000, 1247000, 408

east      = st.sidebar.number_input("LV95 East",       value=east_d,  step=1000)
north     = st.sidebar.number_input("LV95 North",      value=north_d, step=1000)
elevation = st.sidebar.number_input("Höhe (m ü. M.)",  value=elev_d,  step=10)

# ---------- Lagedaten ----------
st.sidebar.header("🌍 Lagedaten")
population = st.sidebar.slider("Bevölkerung Hektar",         1,    600, 100)
oev        = st.sidebar.slider("ÖV-Erschliessung (Score)",   0, 100000, 4000)
solar      = st.sidebar.slider("Solar-Klasse (1=schlecht, 5=top)", 1, 5, 3)

# ---------- Predict ----------
input_df = pd.DataFrame([{
    "east":       east,
    "north":      north,
    "elevation":  elevation,
    "area":       area,
    "rooms":      rooms,
    "year_built": year_built,
    "apartments": apartments,
    "land_area":  land_area,
    "population": population,
    "oev":        oev,
    "solar":      solar,
}])

predicted = float(predictor.predict(input_df)[0])

# ---------- Anzeige ----------
col1, col2, col3 = st.columns(3)
col1.metric("Geschätzte Kaltmiete", f"{predicted:,.0f} CHF".replace(",", "'"))
col2.metric("Pro m²",               f"{predicted / area:.0f} CHF/m²")
col3.metric("Standort",             preset.split(" (")[0])

st.divider()

# ---------- Sensitivität: was kostet ±10 m²? ----------
st.subheader("📊 Sensitivität: Wie ändert sich der Preis?")
ax_col1, ax_col2 = st.columns(2)

with ax_col1:
    # Variiere area
    areas = np.linspace(max(20, area - 30), area + 30, 50)
    df_area = pd.concat([input_df.assign(area=a) for a in areas], ignore_index=True)
    preds_area = predictor.predict(df_area)
    chart_area = pd.DataFrame({"area (m²)": areas, "Predicted Price (CHF)": preds_area})
    st.line_chart(chart_area.set_index("area (m²)"))
    st.caption("Preis-Verlauf bei variierender Wohnfläche (alle anderen Features fix).")

with ax_col2:
    # Variiere rooms
    rooms_range = list(range(max(1, rooms - 3), min(8, rooms + 3) + 1))
    df_rooms = pd.concat([input_df.assign(rooms=r) for r in rooms_range], ignore_index=True)
    preds_rooms = predictor.predict(df_rooms)
    chart_rooms = pd.DataFrame({"rooms": rooms_range, "Predicted Price (CHF)": preds_rooms})
    st.bar_chart(chart_rooms.set_index("rooms"))
    st.caption("Preis-Verlauf bei variierender Zimmerzahl.")

st.divider()

# ---------- Eingabe-Daten ----------
with st.expander("🔍 Eingabe-Daten anzeigen"):
    st.dataframe(input_df.T.rename(columns={0: "Wert"}), use_container_width=True)

with st.expander("ℹ️ Über das Modell"):
    st.markdown(f"""
- **Pipeline:** `RentPredictor` aus `model_v3_clean.ipynb`, Kapitel 24.2
- **Modell:** `{type(predictor.model).__name__}`
- **Features:** {len(predictor.feature_cols)} (inkl. Geo-Cluster & KNN-Distance-Features)
- **Trainings-Daten:** ~4'500 Schweizer Mietwohnungen aus `model.csv`
- **Random State:** {predictor.random_state}

**Wichtig:** Diese Schätzung ist **nicht rechtsverbindlich**.
Sie ist als Marktanalyse-Tool gedacht, nicht als Gutachten oder Mietzins-Berechnung.
    """)
