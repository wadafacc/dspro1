"""Streamlit Demo-App für die Mietpreis-Vorhersage.

Start:
    streamlit run src/app.py

Beim ersten Start wird das Modell aus der CSV-Datei trainiert und gecached.
Folgende Aufrufe lesen das gecachte Modell direkt — das macht die UI schnell.

Pipeline-Erweiterung:
- Adresse → EGID/Koordinaten (geo.admin SearchServer)
- EGID → GWR-Gebäudedaten (gbauj, ganzwhg, garea)
- Koordinaten → Swisstopo (Höhe, ÖV-Score, Solar, Bevölkerung)
- Bereinigung wie in final_records.ipynb
- Predict via bestehender RentPredictor-Klasse (unverändert)
"""
from __future__ import annotations

import datetime as dt
import math
import re
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
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
# Pipeline-Funktionen: Adresse -> EGID -> GWR -> Swisstopo -> Features
# Logik 1:1 aus den drei Notebooks:
#   - gwr_egid_db_sync.ipynb         (search_address, fetch_gwr_feature)
#   - swisstopo_enrich_db_sync_v2    (geocode, get_elevation, identify, parse_*)
#   - final_records.ipynb            (Pflichtfelder + Spalten-Reihenfolge)
# ===========================================================================

# API-Endpunkte (identisch zu den Notebooks)
API_SEARCH_URL   = "https://api3.geo.admin.ch/rest/services/api/SearchServer"
API_FIND_URL     = "https://api3.geo.admin.ch/rest/services/api/MapServer/find"
API_IDENTIFY_URL = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
API_HEIGHT_URL   = "https://api3.geo.admin.ch/rest/services/height"
API_BASE         = "https://api3.geo.admin.ch"

# Offizielle/öffentliche XML-Quellen für EGID-Abfragen.
# 1) Direkter XML-Download der Housing-Stat EGID-Abfrage
# 2) Fallback: MADD/eCH-0206 Webservice
HOUSING_STAT_EGID_XML_URL = "https://www.housing-stat.ch/de/data/query/egid.xml"
MADD_ECH_API_URL          = "https://madd.bfs.admin.ch/eCH-0206"

IDENTIFY_LAYERS = "all:" + ",".join([
    "ch.are.erreichbarkeit-oev",
    "ch.bfe.solarenergie-eignung-daecher",
])
POP_LAYER = "all:ch.bfs.volkszaehlung-bevoelkerungsstatistik_einwohner"

# Spaltenreihenfolge der Modell-CSV (final_records.ipynb FINAL_DATASET_QUERY)
RAW_MODEL_COLUMNS = [
    "area_sqm", "rooms", "population", "oev_score",
    "solar_class", "elevation_m", "lv95_east", "lv95_north",
    "gbauj", "ganzwhg", "garea",
]

# Pflichtspalten (status='usable' aus final_records.ipynb)
REQUIRED_NON_NULL = [
    "address", "area_sqm", "rooms", "population", "egid",
    "gbauj", "ganzwhg", "garea", "oev_score", "solar_class", "elevation_m",
]


def normalize_address(addr: Any) -> str:
    """Whitespace normalisieren — identisch zu den Notebooks."""
    if not isinstance(addr, str):
        return ""
    return re.sub(r"\s+", " ", addr.strip())


def _safe_num(x, default=None) -> Optional[float]:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(x, default=None) -> Optional[int]:
    v = _safe_num(x, None)
    if v is None:
        return default
    try:
        return int(round(v))
    except Exception:
        return default


def _request_json(url: str, params: dict, timeout: int = 15) -> Optional[dict]:
    """HTTP-Wrapper mit einem Retry — analog zu _request/_get_json in den Notebooks."""
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return None
            return None
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                time.sleep(2)
        except requests.exceptions.Timeout:
            if attempt == 0:
                time.sleep(1)
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------
# Public Pipeline-API (Funktionsnamen wie vom User gefordert)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def lookup_egid(address: str) -> Dict[str, Any]:
    """Adresse → EGID + Koordinaten + GWR-Link.

    Aus gwr_egid_db_sync.ipynb (search_address): GeoAdmin SearchServer mit
    origins=address. Parsed featureId zu EGID, sammelt Koordinaten und
    optional einen direkten gwr_link für load_gwr_data.

    Raises
    ------
    ValueError: Adresse leer oder kein Treffer.
    """
    addr = normalize_address(address)
    if not addr:
        raise ValueError("Adresse ist leer.")

    data = _request_json(API_SEARCH_URL, {
        "searchText": addr,
        "type":       "locations",
        "origins":    "address",
        "sr":         2056,
        "limit":      1,
    })
    if not data:
        raise requests.exceptions.RequestException("API-Timeout beim SearchServer.")
    if not data.get("results"):
        raise ValueError(f"Adresse nicht gefunden: '{addr}'")

    best = data["results"][0]
    attrs = best.get("attrs", {}) or {}

    # SearchServer: y=East, x=North (LV95)
    east  = _safe_num(attrs.get("y"))
    north = _safe_num(attrs.get("x"))

    # GWR-Link aus den Result-Links extrahieren
    gwr_link = None
    for link in attrs.get("links", []) or []:
        if link.get("title") == "ch.bfs.gebaeude_wohnungs_register":
            href = link.get("href")
            if href:
                gwr_link = href
                break

    # EGID aus featureId (Format: <EGID>_<EWID>)
    feature_id = attrs.get("featureId") or attrs.get("feature_id")
    egid = None
    if feature_id:
        try:
            egid = int(str(feature_id).split("_")[0])
        except (ValueError, IndexError):
            pass

    if east is None or north is None:
        raise ValueError(f"Keine Koordinaten für '{addr}' verfügbar.")

    label_clean = (attrs.get("label", addr) or addr).replace("<b>", "").replace("</b>", "")

    return {
        "address":     label_clean,
        "egid":        egid,
        "lv95_east":   east,
        "lv95_north":  north,
        "gwr_link":    gwr_link,
        "feature_id":  feature_id,
    }


@st.cache_data(show_spinner=False)
def load_gwr_data(
    egid: Optional[int] = None,
    gwr_link: Optional[str] = None,
) -> Dict[str, Any]:
    """EGID → GWR-Gebäudeattribute (gbauj, ganzwhg, garea, ...).

    Wichtig: `gbauj` ist das Baujahr des Gebäudes. Es wird bewusst nicht aus
    `yearOfConstruction` der einzelnen Wohnung überschrieben.
    """
    if not egid and not gwr_link:
        raise ValueError("Weder EGID noch gwr_link angegeben.")

    feature = None

    # 1) GeoAdmin-Link aus lookup_egid bevorzugen
    if gwr_link:
        url = gwr_link if gwr_link.startswith("http") else f"{API_BASE}{gwr_link}"
        data = _request_json(url, {"returnGeometry": "false"}, timeout=20)
        if data:
            feature = data.get("feature", data)

    # 2) GeoAdmin Find by EGID
    if feature is None and egid is not None:
        data = _request_json(API_FIND_URL, {
            "layer":          "ch.bfs.gebaeude_wohnungs_register",
            "searchText":     str(egid),
            "searchField":    "egid",
            "returnGeometry": "false",
        }, timeout=20)
        if data and data.get("results"):
            feature = data["results"][0]

    attrs: Dict[str, Any] = {}
    if feature is not None:
        attrs_raw = feature.get("attributes", {}) or {}
        attrs = {str(k).lower(): v for k, v in attrs_raw.items()}

    # Werte aus GeoAdmin, falls vorhanden
    result = {
        "egid":     _safe_int(attrs.get("egid", egid)),
        "gbauj":    _safe_int(attrs.get("gbauj")),
        "gbaup":    _safe_int(attrs.get("gbaup")),
        "ganzwhg":  _safe_int(attrs.get("ganzwhg")),
        "garea":    _safe_num(attrs.get("garea")),
        "_attrs":   attrs,
    }

    # 3) Fallback/Ergänzung aus MADD-XML.
    #    Hier kommt das Baujahr aus building/dateOfConstruction,
    #    nicht aus dwelling/yearOfConstruction.
    if egid is not None and (
        result["gbauj"] is None
        or result["ganzwhg"] is None
        or result["garea"] is None
    ):
        try:
            xml_text, madd_debug = _fetch_madd_xml_for_egid(int(egid), timeout=20)
            if xml_text:
                building = _parse_building_from_madd_xml(xml_text)
                if result["egid"] is None:
                    result["egid"] = _safe_int(building.get("egid"), egid)
                if result["gbauj"] is None:
                    result["gbauj"] = _safe_int(building.get("gbauj"))
                if result["ganzwhg"] is None:
                    result["ganzwhg"] = _safe_int(building.get("ganzwhg"))
                if result["garea"] is None:
                    result["garea"] = _safe_num(building.get("garea"))
                result["_madd_debug"] = madd_debug
        except Exception as exc:
            result["_madd_error"] = f"{type(exc).__name__}: {exc}"

    if (
        result["gbauj"] is None
        and result["ganzwhg"] is None
        and result["garea"] is None
        and feature is None
    ):
        raise ValueError(f"GWR-Daten nicht gefunden (egid={egid}).")

    return result

# GWR-WSTWK Stockwerk-Codes (BFS-Standard)
WSTWK_CODE_MAP: Dict[int, str] = {
    3100: "Sockelgeschoss",
    3300: "UG",
    3401: "1. UG",
    3402: "2. UG",
    3403: "3. UG",
    3413: "EG",
    3500: "EG",
    3501: "1. OG",
    3502: "2. OG",
    3503: "3. OG",
    3504: "4. OG",
    3505: "5. OG",
    3506: "6. OG",
    3507: "7. OG",
    3508: "8. OG",
    3601: "1. DG",
    3602: "2. DG",
}


def parse_gwr_floor(raw_floor: Any) -> Optional[str]:
    """Heuristisches Parsing der Stockwerk-Codes aus GWR/MADD.

    MADD/eCH liefert für Wohnungen teilweise Codes wie 3101, 3102, ...
    Diese werden als 1. Stock, 2. Stock usw. angezeigt.
    """
    if raw_floor is None or raw_floor == "":
        return None

    s = str(raw_floor).strip()
    if not s:
        return None

    code = _safe_int(s)
    if code is None:
        return s

    # Wichtig für MADD XML: 3101, 3102, 3103, ...
    if 3101 <= code <= 3199:
        return f"{code - 3100}. Stock"

    if code in WSTWK_CODE_MAP:
        return WSTWK_CODE_MAP[code]
    if code == 0:
        return "EG"
    if 1 <= code <= 20:
        return f"{code}. OG"
    if code < 0:
        return f"{abs(code)}. UG"

    return str(code)

APARTMENT_FIELD_KEYS = (
    "ewid", "wstwk", "wflae", "wazim", "wbez", "wstat",
    "stockwerk", "flaeche", "zimmer", "bezeichnung",
    "floor", "area", "rooms", "label",
    "administrativedwellingno", "noofhabitablerooms", "surfaceareaofdwelling",
)


def _is_scalar(v) -> bool:
    """True falls v ein einzelner Skalar ist (keine Liste, kein Dict)."""
    return v is not None and not isinstance(v, (list, dict, tuple, set))


def _is_dwelling_record(d: dict) -> bool:
    """Heuristik: dict sieht wie eine einzelne Wohnung aus.

    Lenient: mindestens 2 Wohnungs-Felder als Skalar.  Wir verlangen *nicht*
    explizit EWID, weil manche APIs EWID weglassen oder anders benennen.
    Zwei Skalar-Felder verhindern Aggregat-Wrapper (wo Felder als Listen
    vorliegen) als False-Positives.
    """
    if not isinstance(d, dict):
        return False
    klow = {str(k).lower(): v for k, v in d.items()}
    n_scalar = sum(1 for k in APARTMENT_FIELD_KEYS if _is_scalar(klow.get(k)))
    return n_scalar >= 2


def _extract_dwelling(d: dict) -> dict:
    klow = {str(k).lower(): v for k, v in d.items()}

    def pick(*keys):
        for k in keys:
            v = klow.get(k)
            if _is_scalar(v):
                return v
        return None

    floor_raw = pick("wstwk", "stockwerk", "floor")
    return {
        "ewid":        _safe_int(pick("ewid")),
        "label":       pick("wbez", "bezeichnung", "label", "administrativedwellingno"),
        "floor_raw":   floor_raw,
        "floor_label": parse_gwr_floor(floor_raw),
        "rooms":       _safe_num(pick("wazim", "zimmer", "rooms", "noofhabitablerooms")),
        "area":        _safe_num(pick("wflae", "flaeche", "area", "surfaceareaofdwelling")),
    }


def _walk_dwellings(obj, seen: set) -> list:
    """Rekursive Suche nach Wohnungs-Records, dedupliziert über EWID oder Field-Tuple."""
    out: list = []
    if isinstance(obj, dict):
        if _is_dwelling_record(obj):
            d = _extract_dwelling(obj)
            # Dedup-Key: EWID falls vorhanden, sonst Tuple aus charakteristischen Feldern
            ewid = d.get("ewid")
            dedup_key = ("ewid", ewid) if ewid else (
                "tup", d.get("floor_label"), d.get("rooms"),
                d.get("area"), d.get("label"),
            )
            if dedup_key not in seen:
                seen.add(dedup_key)
                out.append(d)
            # Nicht weiter in einen Dwelling-Record reinrekursieren
            return out
        for v in obj.values():
            out.extend(_walk_dwellings(v, seen))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(_walk_dwellings(item, seen))
    return out


def _parse_dwellings_from_html(html: str) -> list:
    """Heuristisch HTML-Tabelle aus GeoAdmin htmlPopup parsen.

    GeoAdmin liefert für Gebäude eine HTML-Tabelle der Wohnungen. Wir suchen
    nach <tr>-Zeilen, identifizieren die Header (EWID/Stockwerk/Zimmer/Fläche)
    und mappen die Daten.
    """
    out: list = []
    if not html or len(html) < 50:
        return out

    # Finde alle <tr>-Zeilen
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL)
    if not rows:
        return out

    def _strip(cell: str) -> str:
        cell = re.sub(r"<[^>]+>", "", cell)
        cell = cell.replace("&nbsp;", " ").replace("&amp;", "&")
        return cell.strip()

    # Erste Tabelle mit Header identifizieren
    field_map: Dict[str, int] = {}
    data_rows: list = []
    for row in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row,
                           re.IGNORECASE | re.DOTALL)
        cells = [_strip(c) for c in cells]
        if not cells:
            continue
        # Header-Zeile?
        if not field_map:
            row_lower = " ".join(c.lower() for c in cells)
            looks_like_header = any(k in row_lower for k in
                                     ("ewid", "stockwerk", "wstwk", "wbez",
                                      "zimmer", "wazim", "fläche", "wflae"))
            if looks_like_header:
                for idx, col in enumerate(cells):
                    cl = col.lower()
                    if "ewid" in cl:
                        field_map["ewid"] = idx
                    elif "stockwerk" in cl or "wstwk" in cl or "etage" in cl:
                        field_map["floor_raw"] = idx
                    elif "zimmer" in cl or "wazim" in cl:
                        field_map["rooms"] = idx
                    elif "fläche" in cl or "flaeche" in cl or "wflae" in cl:
                        field_map["area"] = idx
                    elif "bezeich" in cl or "wbez" in cl or "wohnung" in cl:
                        field_map["label"] = idx
                continue
        if field_map:
            data_rows.append(cells)

    if not field_map or not data_rows:
        return out

    for row in data_rows:
        d = {"ewid": None, "label": None, "floor_raw": None,
             "floor_label": None, "rooms": None, "area": None}
        try:
            if "ewid" in field_map and field_map["ewid"] < len(row):
                d["ewid"] = _safe_int(row[field_map["ewid"]])
            if "label" in field_map and field_map["label"] < len(row):
                v = row[field_map["label"]]
                d["label"] = v if v else None
            if "floor_raw" in field_map and field_map["floor_raw"] < len(row):
                v = row[field_map["floor_raw"]]
                d["floor_raw"]   = v
                d["floor_label"] = parse_gwr_floor(v)
            if "rooms" in field_map and field_map["rooms"] < len(row):
                v = row[field_map["rooms"]].replace(",", ".")
                d["rooms"] = _safe_num(re.sub(r"[^\d.\-]", "", v))
            if "area" in field_map and field_map["area"] < len(row):
                v = row[field_map["area"]].replace(",", ".")
                d["area"] = _safe_num(re.sub(r"[^\d.\-]", "", v))
        except Exception:
            continue
        # Mindestens ein verwertbares Feld
        if any(d.get(k) is not None for k in ("ewid", "rooms", "area", "floor_label", "label")):
            out.append(d)

    return out


# --------------------------------------------------------------------------
# MADD / eCH-0206 XML: EGID -> Gebäude + Wohnungen
# --------------------------------------------------------------------------
def _xml_local_name(tag: str) -> str:
    """Entfernt XML-Namespaces: '{namespace}EGID' -> 'EGID'."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _xml_direct_child(parent, name: str):
    """Direktes Kind-Element nach lokalem Namen suchen."""
    if parent is None:
        return None
    name_l = name.lower()
    for child in list(parent):
        if _xml_local_name(child.tag).lower() == name_l:
            return child
    return None


def _xml_first_text(parent, *names: str) -> Optional[str]:
    """Rekursiv ersten Text für einen lokalen XML-Namen finden."""
    if parent is None:
        return None
    wanted = {n.lower() for n in names}
    for el in parent.iter():
        if _xml_local_name(el.tag).lower() in wanted:
            txt = (el.text or "").strip()
            if txt:
                return txt
    return None


def _looks_like_xml(text: str) -> bool:
    s = (text or "").lstrip()
    return s.startswith("<?xml") or s.startswith("<maddResponse") or "<maddResponse" in s[:500]


def _build_madd_request_xml(egid: int) -> str:
    """Minimale eCH-0206 maddRequest-Anfrage für EGID/building."""
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    msg_id = str(uuid.uuid4())
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<eCH-0206:maddRequest
    xmlns:eCH-0206="http://www.ech.ch/xmlns/eCH-0206/2"
    xmlns:eCH-0058="http://www.ech.ch/xmlns/eCH-0058/5"
    xmlns:eCH-0129="http://www.ech.ch/xmlns/eCH-0129/5"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <eCH-0206:requestHeader>
    <eCH-0206:messageId>{msg_id}</eCH-0206:messageId>
    <eCH-0206:businessReferenceId>{msg_id}</eCH-0206:businessReferenceId>
    <eCH-0206:requestingApplication>
      <eCH-0058:manufacturer>HSLU</eCH-0058:manufacturer>
      <eCH-0058:product>RentPredictorStreamlit</eCH-0058:product>
      <eCH-0058:productVersion>1.0</eCH-0058:productVersion>
    </eCH-0206:requestingApplication>
    <eCH-0206:requestDate>{now}</eCH-0206:requestDate>
  </eCH-0206:requestHeader>
  <eCH-0206:requestContext>building</eCH-0206:requestContext>
  <eCH-0206:requestQuery>
    <eCH-0206:EGID>{int(egid)}</eCH-0206:EGID>
  </eCH-0206:requestQuery>
</eCH-0206:maddRequest>'''


def _fetch_madd_xml_for_egid(egid: int, timeout: int = 20) -> tuple[str, Dict[str, Any]]:
    """Automatischer XML-Abruf für eine EGID.

    Reihenfolge:
    1. Direkter XML-Download der Housing-Stat EGID-Abfrage.
    2. Fallback: MADD/eCH-0206 als POST mit XML-Body.
    3. Fallback: MADD/eCH-0206 als GET mit egid-Parameter.
    """
    headers_xml = {
        "Accept": "application/xml,text/xml,*/*",
        "User-Agent": "rent-predictor-streamlit/1.0",
    }
    attempts: list[Dict[str, Any]] = []

    # 1) Direkter XML-Endpunkt passend zur EGID-Webabfrage
    try:
        r = requests.get(
            HOUSING_STAT_EGID_XML_URL,
            params={"egid": int(egid)},
            headers=headers_xml,
            timeout=timeout,
        )
        info = {
            "method": "GET",
            "url": r.url,
            "status": r.status_code,
            "content_type": r.headers.get("Content-Type"),
            "text_start": r.text[:160],
        }
        attempts.append(info)
        if r.status_code == 200 and _looks_like_xml(r.text):
            return r.text, {"success": True, "used": info, "attempts": attempts}
    except Exception as exc:
        attempts.append({
            "method": "GET",
            "url": HOUSING_STAT_EGID_XML_URL,
            "exception": f"{type(exc).__name__}: {exc}",
        })

    # 2) MADD/eCH-0206 POST
    xml_body = _build_madd_request_xml(egid)
    try:
        r = requests.post(
            MADD_ECH_API_URL,
            data=xml_body.encode("utf-8"),
            headers={
                "Content-Type": "application/xml; charset=utf-8",
                "Accept": "application/xml,text/xml,*/*",
                "User-Agent": "rent-predictor-streamlit/1.0",
            },
            timeout=timeout,
        )
        info = {
            "method": "POST",
            "url": r.url,
            "status": r.status_code,
            "content_type": r.headers.get("Content-Type"),
            "text_start": r.text[:160],
        }
        attempts.append(info)
        if r.status_code == 200 and _looks_like_xml(r.text):
            return r.text, {"success": True, "used": info, "attempts": attempts}
    except Exception as exc:
        attempts.append({
            "method": "POST",
            "url": MADD_ECH_API_URL,
            "exception": f"{type(exc).__name__}: {exc}",
        })

    # 3) MADD/eCH-0206 GET
    try:
        r = requests.get(
            MADD_ECH_API_URL,
            params={"egid": int(egid)},
            headers=headers_xml,
            timeout=timeout,
        )
        info = {
            "method": "GET",
            "url": r.url,
            "status": r.status_code,
            "content_type": r.headers.get("Content-Type"),
            "text_start": r.text[:160],
        }
        attempts.append(info)
        if r.status_code == 200 and _looks_like_xml(r.text):
            return r.text, {"success": True, "used": info, "attempts": attempts}
    except Exception as exc:
        attempts.append({
            "method": "GET",
            "url": MADD_ECH_API_URL,
            "exception": f"{type(exc).__name__}: {exc}",
        })

    return "", {"success": False, "attempts": attempts}


def _parse_building_from_madd_xml(xml_text: str) -> Dict[str, Any]:
    """Gebäudeattribute aus MADD XML lesen.

    `gbauj` stammt aus building/dateOfConstruction/dateOfConstruction.
    `ganzwhg` wird aus der Anzahl dwellingItem gelesen.
    `garea` entspricht surfaceAreaOfBuilding.
    """
    root = ET.fromstring(xml_text)

    dwelling_count = sum(
        1 for el in root.iter()
        if _xml_local_name(el.tag) == "dwellingItem"
    )

    for building_item in root.iter():
        if _xml_local_name(building_item.tag) != "buildingItem":
            continue

        building = _xml_direct_child(building_item, "building")
        if building is None:
            continue

        date_node = _xml_direct_child(building, "dateOfConstruction")

        return {
            "egid":     _safe_int(_xml_first_text(building_item, "EGID")),
            "gbauj":    _safe_int(_xml_first_text(date_node, "dateOfConstruction")),
            "gbaup":    _safe_int(_xml_first_text(date_node, "periodOfConstruction")),
            "ganzwhg":  dwelling_count or None,
            "garea":    _safe_num(_xml_first_text(building, "surfaceAreaOfBuilding")),
            "building_status":   _safe_int(_xml_first_text(building, "buildingStatus")),
            "building_category": _safe_int(_xml_first_text(building, "buildingCategory")),
            "building_class":    _safe_int(_xml_first_text(building, "buildingClass")),
            "number_of_floors":  _safe_int(_xml_first_text(building, "numberOfFloors")),
        }

    return {
        "egid":     None,
        "gbauj":    None,
        "gbaup":    None,
        "ganzwhg":  dwelling_count or None,
        "garea":    None,
    }


def _parse_dwellings_from_madd_xml(xml_text: str) -> list:
    """Wohnungen aus MADD/eCH-0206 XML parsen.

    Das Wohnungs-Baujahr wird bewusst als `year_built_dwelling` geführt und
    nicht als Modell-Baujahr verwendet. Für das Modell bleibt `gbauj`
    aus dem Gebäude massgebend.
    """
    out: list = []
    seen: set = set()

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out

    for item in root.iter():
        if _xml_local_name(item.tag) != "dwellingItem":
            continue

        ewid = _safe_int(_xml_first_text(item, "EWID"))
        admin_no = _xml_first_text(item, "administrativeDwellingNo")
        floor_raw = _xml_first_text(item, "floor")

        d = {
            "ewid":                ewid,
            "label":               admin_no,
            "floor_raw":           _safe_int(floor_raw, floor_raw),
            "floor_label":         parse_gwr_floor(floor_raw),
            "rooms":               _safe_num(_xml_first_text(item, "noOfHabitableRooms")),
            "area":                _safe_num(_xml_first_text(item, "surfaceAreaOfDwelling")),
            "year_built_dwelling": _safe_int(_xml_first_text(item, "yearOfConstruction")),
            "kitchen":             _safe_int(_xml_first_text(item, "kitchen")),
            "dwelling_status":     _safe_int(_xml_first_text(item, "dwellingStatus")),
        }

        key = ("ewid", ewid) if ewid else (
            "tup", d.get("floor_label"), d.get("rooms"),
            d.get("area"), d.get("label"),
        )
        if key not in seen:
            seen.add(key)
            out.append(d)

    return out


@st.cache_data(show_spinner=False)
def load_gwr_dwellings_with_debug(egid: int) -> tuple:
    """EGID → (Wohnungsliste, Debug-Info).

    Primär wird automatisch die XML/API-Antwort von Housing-Stat/MADD gelesen.
    Die alten GeoAdmin-Varianten bleiben nur als Fallback drin.
    """
    debug: Dict[str, Any] = {"egid": egid, "attempts": []}
    if not egid:
        return [], debug

    seen: set = set()
    out: list = []

    # === 1. MADD / Housing-Stat XML: echte Wohnungsdetails ===
    try:
        xml_text, madd_debug = _fetch_madd_xml_for_egid(int(egid), timeout=20)
        attempt = {
            "source": "housing-stat / MADD XML",
            "success": bool(xml_text),
            **madd_debug,
        }
        if xml_text:
            parsed = _parse_dwellings_from_madd_xml(xml_text)
            attempt["xml_len"] = len(xml_text)
            attempt["dwellings_parsed"] = len(parsed)

            for d in parsed:
                ewid = d.get("ewid")
                key = ("ewid", ewid) if ewid else (
                    "tup", d.get("floor_label"), d.get("rooms"),
                    d.get("area"), d.get("label"),
                )
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        debug["attempts"].append(attempt)
    except Exception as exc:
        debug["attempts"].append({
            "source": "housing-stat / MADD XML",
            "exception": f"{type(exc).__name__}: {exc}",
        })

    headers = {
        "Accept": "application/json",
        "User-Agent": "rent-predictor-streamlit/1.0",
    }

    # === 2. GeoAdmin Feature-Endpoint (Fallback) ===
    if not out:
        feat_url = (
            f"https://api3.geo.admin.ch/rest/services/ech/MapServer/"
            f"ch.bfs.gebaeude_wohnungs_register/{int(egid)}"
            f"?returnGeometry=false&lang=de"
        )
        attempt = {"source": "geo.admin.ch feature", "url": feat_url}
        try:
            r = requests.get(feat_url, headers=headers, timeout=20)
            attempt["status"] = r.status_code
            if r.status_code == 200:
                try:
                    data = r.json()
                    if isinstance(data, dict):
                        attempt["top_keys"] = list(data.keys())[:10]
                    before = len(out)
                    out.extend(_walk_dwellings(data, seen))
                    attempt["dwellings_found"] = len(out) - before
                except Exception as e:
                    attempt["json_error"] = str(e)
        except Exception as e:
            attempt["exception"] = f"{type(e).__name__}: {e}"
        debug["attempts"].append(attempt)

    # === 3. GeoAdmin htmlPopup (Fallback) ===
    if not out:
        for popup_kind in ("extendedHtmlPopup", "htmlPopup"):
            popup_url = (
                f"https://api3.geo.admin.ch/rest/services/ech/MapServer/"
                f"ch.bfs.gebaeude_wohnungs_register/{int(egid)}/{popup_kind}"
                f"?lang=de"
            )
            attempt = {"source": f"geo.admin.ch {popup_kind}", "url": popup_url}
            try:
                r = requests.get(
                    popup_url,
                    headers={"Accept": "text/html",
                             "User-Agent": headers["User-Agent"]},
                    timeout=20,
                )
                attempt["status"] = r.status_code
                if r.status_code == 200:
                    parsed = _parse_dwellings_from_html(r.text)
                    attempt["html_len"] = len(r.text)
                    attempt["dwellings_parsed"] = len(parsed)
                    for d in parsed:
                        ewid = d.get("ewid")
                        key = ("ewid", ewid) if ewid else (
                            "tup", d.get("floor_label"), d.get("rooms"),
                            d.get("area"), d.get("label"),
                        )
                        if key not in seen:
                            seen.add(key)
                            out.append(d)
            except Exception as e:
                attempt["exception"] = f"{type(e).__name__}: {e}"
            debug["attempts"].append(attempt)
            if out:
                break

    # === 4. GeoAdmin Find (letzter Fallback) ===
    if not out:
        attempt = {"source": "geo.admin.ch find", "egid": egid}
        try:
            data = _request_json(API_FIND_URL, {
                "layer":          "ch.bfs.gebaeude_wohnungs_register",
                "searchText":     str(int(egid)),
                "searchField":    "egid",
                "returnGeometry": "false",
                "limit":          50,
            }, timeout=20)
            attempt["got_data"] = data is not None
            if data is not None:
                if isinstance(data, dict):
                    attempt["top_keys"] = list(data.keys())[:10]
                before = len(out)
                out.extend(_walk_dwellings(data, seen))
                attempt["dwellings_found"] = len(out) - before
        except Exception as e:
            attempt["exception"] = f"{type(e).__name__}: {e}"
        debug["attempts"].append(attempt)

    # Sortierung: EWID zuerst, sonst Stockwerk/Label
    def _sort_key(d):
        ewid = d.get("ewid")
        if ewid is not None:
            return (0, ewid)
        fr = d.get("floor_raw")
        if isinstance(fr, (int, float)):
            return (1, fr)
        if isinstance(fr, str) and fr.strip():
            return (2, fr)
        return (3, "")

    out.sort(key=_sort_key)
    debug["total_dwellings"] = len(out)
    return out, debug
def make_manual_entry_dwelling(area_default: float = 75.0,
                                 rooms_default: float = 3.0) -> dict:
    """Letzter Fallback: ein einziger generischer Eintrag für manuelle Eingabe.

    Wird verwendet, wenn weder die GWR-API noch die Synthese aus ganzwhg
    Wohnungen liefern. Damit hat das Dropdown immer mindestens einen Eintrag
    und der Nutzer kann Fläche / Zimmer / Stockwerk frei eingeben.
    """
    return {
        "ewid":        None,
        "label":       "Manuelle Eingabe (keine GWR-Daten verfügbar)",
        "floor_raw":   None,
        "floor_label": None,
        "rooms":       rooms_default,
        "area":        area_default,
    }


def synthesize_dwellings_from_building(gwr_building: Dict[str, Any]) -> list:
    """Fallback: aus `ganzwhg` und `garea` synthetische Wohnungs-Einträge bauen.

    Wenn keine echte API Wohnungen liefert, generieren wir N generische Einträge,
    bei denen `area` der Gebäude-Durchschnitt ist (`garea / ganzwhg`).
    Stockwerk und Zimmer bleiben offen, der Nutzer trägt sie manuell nach.
    """
    if not gwr_building:
        return []
    ganzwhg = _safe_int(gwr_building.get("ganzwhg"))
    garea   = _safe_num(gwr_building.get("garea"))
    if not ganzwhg or ganzwhg <= 0:
        return []
    avg_area = (garea / ganzwhg) if (garea and ganzwhg) else None
    return [
        {
            "ewid":        None,
            "label":       f"Wohnung {i+1} (Schätzung aus GWR-Total)",
            "floor_raw":   None,
            "floor_label": None,
            "rooms":       None,
            "area":        avg_area,
        }
        for i in range(int(ganzwhg))
    ]


def load_gwr_dwellings(egid: int) -> list:
    """Thin wrapper, nur die Liste — für bestehende Aufrufer."""
    out, _ = load_gwr_dwellings_with_debug(egid)
    return out


@st.cache_data(show_spinner=False)
def load_swisstopo_data(east: float, north: float) -> Dict[str, Any]:
    """LV95-Koordinaten → Höhe + ÖV-Score + Solar-Klasse + Bevölkerung.

    Aus swisstopo_enrich_db_sync_v2.ipynb (enrich_address ohne den
    geocode-Step, der schon in lookup_egid passiert ist).
    """
    # Höhe ü. M.
    h = _request_json(API_HEIGHT_URL, {"easting": east, "northing": north}, timeout=15)
    elevation = _safe_num((h or {}).get("height"))

    # ÖV-Score + Solar-Klasse via identify (tolerance=1, erste Treffer)
    ident = _request_json(API_IDENTIFY_URL, {
        "geometry":       f"{east},{north}",
        "geometryType":   "esriGeometryPoint",
        "layers":         IDENTIFY_LAYERS,
        "tolerance":      1,
        "returnGeometry": "false",
        "sr":             2056,
        "imageDisplay":   "100,100,96",
        "mapExtent":      f"{east-10},{north-10},{east+10},{north+10}",
    }, timeout=20)

    oev_score, solar_class = None, None
    for item in (ident or {}).get("results") or []:
        layer = item.get("layerBodId", "")
        attr  = item.get("attributes", {}) or {}
        if oev_score is None and layer == "ch.are.erreichbarkeit-oev":
            oev_score = _safe_num(attr.get("oev_erreichb_ewap"))
        elif solar_class is None and layer == "ch.bfe.solarenergie-eignung-daecher":
            solar_class = _safe_int(attr.get("klasse"))
        if oev_score is not None and solar_class is not None:
            break

    # Bevölkerung — nächstgelegene Hektarzelle
    pop_resp = _request_json(API_IDENTIFY_URL, {
        "geometry":       f"{east},{north}",
        "geometryType":   "esriGeometryPoint",
        "layers":         POP_LAYER,
        "tolerance":      1,
        "returnGeometry": "false",
        "sr":             2056,
        "imageDisplay":   "100,100,96",
        "mapExtent":      f"{east-10},{north-10},{east+10},{north+10}",
    }, timeout=15)

    population = None
    for item in (pop_resp or {}).get("results") or []:
        attr = item.get("attributes", {}) or {}
        n_val = _safe_int(attr.get("number"))
        y_val = attr.get("i_year")
        if n_val is None:
            continue
        if y_val is None or y_val == 2024:
            population = n_val
            break

    return {
        "elevation_m": elevation,
        "oev_score":   oev_score,
        "solar_class": solar_class,
        "population":  population,
    }


def assemble_features(
    *,
    address: str,
    area_sqm: float,
    rooms: float,
    floor: Optional[str] = None,
    egid_info: Dict[str, Any],
    gwr_info: Dict[str, Any],
    swisstopo_info: Dict[str, Any],
) -> pd.DataFrame:
    """Kombiniert alle Quellen zu einer Roh-Zeile mit der Spaltenstruktur
    aus final_records.ipynb (Modell-CSV).

    `floor` wird mitgeführt (Stockwerk), ist aktuell aber kein Modellfeature.
    """
    raw = {
        "address":      address,
        "area_sqm":     _safe_num(area_sqm),
        "rooms":        _safe_num(rooms),
        "lv95_east":    _safe_num(egid_info.get("lv95_east")),
        "lv95_north":   _safe_num(egid_info.get("lv95_north")),
        "egid":         _safe_int(egid_info.get("egid")),
        "gbauj":        _safe_int(gwr_info.get("gbauj")),
        "ganzwhg":      _safe_int(gwr_info.get("ganzwhg")),
        "garea":        _safe_num(gwr_info.get("garea")),
        "elevation_m":  _safe_num(swisstopo_info.get("elevation_m")),
        "oev_score":    _safe_num(swisstopo_info.get("oev_score")),
        "solar_class":  _safe_int(swisstopo_info.get("solar_class")),
        "population":   _safe_int(swisstopo_info.get("population")),
        "floor":        floor,  # informativ
    }
    return pd.DataFrame([raw])


def clean_and_finalize_records(
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Bereinigung + Harmonisierung wie in final_records.ipynb.

    1. Pflichtspalten dürfen nicht NULL sein (Status='usable')
    2. Typkonvertierungen wie im Notebook
    3. Spaltenreihenfolge wie in model.csv
    4. Rename auf Trainings-Spaltennamen (area_sqm → area, etc.)
    """
    df = raw_df.copy()
    status: Dict[str, Any] = {"warnings": [], "missing": []}

    # Status-Check (final_records.ipynb 'usable'-Filter)
    for col in REQUIRED_NON_NULL:
        if col in df.columns and df[col].isna().any():
            status["missing"].append(col)

    if status["missing"]:
        status["warnings"].append(
            f"Pflichtspalten fehlend ('usable'-Filter aus final_records.ipynb): "
            f"{status['missing']}"
        )

    # Typen wie im Notebook
    for c in ["area_sqm", "rooms", "population", "egid",
              "gbauj", "ganzwhg", "solar_class"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["lv95_east", "lv95_north", "elevation_m", "oev_score", "garea"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Spaltenreihenfolge wie in model.csv (FINAL_DATASET_QUERY)
    model_cols_raw = [c for c in RAW_MODEL_COLUMNS if c in df.columns]
    df_model = df[model_cols_raw].copy()

    # Rename auf Trainings-Namen (siehe COLUMN_RENAMES weiter unten)
    df_model = df_model.rename(columns=COLUMN_RENAMES)

    return df_model, status


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

# ===========================================================================
# 🔍 Adress-Lookup (2-stufig: Adresse → Wohnungs-Dropdown → Predict)
# Bestehender manueller Sidebar-Flow unten bleibt unverändert.
# ===========================================================================
st.subheader("🔍 Adress-Lookup mit GeoAdmin & GWR")
st.markdown(
    "Gib eine Adresse ein → wir suchen die EGID, holen alle Wohnungen aus dem "
    "GWR und du wählst eine aus dem Dropdown. Wohnfläche, Zimmerzahl und "
    "Stockwerk werden automatisch übernommen. Du kannst sie noch anpassen, "
    "bevor der bestehende `RentPredictor.predict()` aufgerufen wird."
)

# --- Session State Initialisierung ---
st.session_state.setdefault("lookup_egid_info", None)
st.session_state.setdefault("lookup_dwellings", [])
st.session_state.setdefault("lookup_address",   "Kronenbergstrasse 5, Thalwil")
st.session_state.setdefault("lookup_area",      75)
st.session_state.setdefault("lookup_rooms",     3.0)
st.session_state.setdefault("lookup_floor",     "—")
st.session_state.setdefault("dwelling_selector_idx", 0)


def _apply_dwelling_to_inputs(sel: dict) -> None:
    """Übernimmt Werte einer Wohnung in die Streamlit-Inputs (Session State)."""
    if sel.get("area") is not None:
        try:
            st.session_state.lookup_area = max(10, min(500, int(round(float(sel["area"])))))
        except (ValueError, TypeError):
            pass
    if sel.get("rooms") is not None:
        try:
            st.session_state.lookup_rooms = max(0.5, min(15.0, float(sel["rooms"])))
        except (ValueError, TypeError):
            pass
    fl = sel.get("floor_label")
    if fl:
        st.session_state.lookup_floor = fl


def _on_dwelling_change():
    """Callback: bei Wohnungs-Auswahl Wohnfläche/Zimmer/Stockwerk auto-fill."""
    idx = st.session_state.get("dwelling_selector_idx", 0)
    dwellings = st.session_state.get("lookup_dwellings") or []
    if 0 <= idx < len(dwellings):
        _apply_dwelling_to_inputs(dwellings[idx])

# --- Stage 1: Adresse + "Gebäude suchen" ---
addr_col, search_col = st.columns([7, 2])
address_input = addr_col.text_input(
    "Adresse (Strasse Nr, PLZ Ort)",
    value=st.session_state.lookup_address,
    placeholder="z. B. Limmatquai 28, 8001 Zürich",
    key="lookup_address_input",
)
search_clicked = search_col.button(
    "📍 Gebäude suchen",
    type="primary",
    use_container_width=True,
)

if search_clicked:
    # Streamlit-Cache der Wohnungs-Funktion invalidieren, damit neue Code-Pfade
    # (htmlPopup, Feature-Endpoint) bei Re-Run aktiv werden.
    try:
        load_gwr_dwellings_with_debug.clear()
    except Exception:
        pass

    # Reset: Inputs auf Default, Selector zurück auf 0, alte Debug-Info löschen
    st.session_state.lookup_area  = 75
    st.session_state.lookup_rooms = 3.0
    st.session_state.lookup_floor = "—"
    st.session_state.dwelling_selector_idx = 0
    st.session_state.lookup_dwellings_debug = None
    st.session_state.lookup_dwellings_synthesized = False

    try:
        with st.spinner("Suche Adresse via GeoAdmin SearchServer ..."):
            egid_info = lookup_egid(address_input)

        dwellings: list = []
        debug_info: Dict[str, Any] = {}
        if egid_info.get("egid"):
            with st.spinner(f"Lade alle Wohnungen für EGID {egid_info['egid']} ..."):
                try:
                    dwellings, debug_info = load_gwr_dwellings_with_debug(
                        int(egid_info["egid"])
                    )
                except Exception as exc:
                    st.warning(
                        f"⚠️ Wohnungs-Liste nicht abrufbar — manuelle Eingabe nötig "
                        f"({type(exc).__name__}: {exc})"
                    )
                    debug_info = {"error": f"{type(exc).__name__}: {exc}"}

            # FALLBACK: keine echten Wohnungs-Daten → aus ganzwhg synthetisieren
            if not dwellings:
                try:
                    gwr_building = load_gwr_data(
                        egid=egid_info.get("egid"),
                        gwr_link=egid_info.get("gwr_link"),
                    )
                    synth = synthesize_dwellings_from_building(gwr_building)
                    if synth:
                        dwellings = synth
                        st.session_state.lookup_dwellings_synthesized = True
                        debug_info["synthesized_from_ganzwhg"] = {
                            "ganzwhg": gwr_building.get("ganzwhg"),
                            "garea":   gwr_building.get("garea"),
                            "n_synth": len(synth),
                        }
                except Exception as exc:
                    debug_info["synthesize_error"] = f"{type(exc).__name__}: {exc}"

            # Letzter Fallback: garantiere mindestens einen Eintrag,
            # damit die UI nicht in einen leeren-Zustand kippt.
            if not dwellings and egid_info.get("egid"):
                dwellings = [make_manual_entry_dwelling()]
                st.session_state.lookup_dwellings_synthesized = True
                debug_info["fallback_manual_entry"] = True

        st.session_state.lookup_egid_info       = egid_info
        st.session_state.lookup_dwellings       = dwellings
        st.session_state.lookup_dwellings_debug = debug_info
        st.session_state.lookup_address         = address_input
        # Auto-Fill mit der ersten Wohnung, falls vorhanden
        if dwellings:
            _apply_dwelling_to_inputs(dwellings[0])
    except ValueError as e:
        st.session_state.lookup_egid_info = None
        st.session_state.lookup_dwellings = []
        st.error(f"❌ {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"⏱️ API-Timeout / Verbindungsfehler: {e}")
    except Exception as e:
        st.error(f"Unerwarteter Fehler: {type(e).__name__}: {e}")


# --- Stage 2: Wohnung auswählen + Predict ---
egid_info_state = st.session_state.lookup_egid_info
dwellings_state = st.session_state.lookup_dwellings

if egid_info_state is not None:
    n_dw = len(dwellings_state)
    is_synth = st.session_state.get("lookup_dwellings_synthesized", False)
    msg = (
        f"✅ Gefunden: **{egid_info_state['address']}**  ·  "
        f"EGID `{egid_info_state.get('egid', '—')}`  ·  "
    )
    is_manual_only = (
        is_synth and n_dw == 1
        and dwellings_state and dwellings_state[0].get("label", "").startswith("Manuelle Eingabe")
    )
    if n_dw > 0 and not is_synth:
        msg += f"**{n_dw} Wohnung(en)** im Gebäude verfügbar"
    elif is_manual_only:
        msg += "**1 Standard-Eintrag** (Manuelle Werte unten anpassen)"
    elif n_dw > 0 and is_synth:
        msg += f"**{n_dw} Wohnung(en) (aus GWR-Total geschätzt)**"
    elif egid_info_state.get("egid"):
        msg += "Wohnungs-Liste leer, Werte manuell eingeben"
    else:
        msg += "Keine EGID gefunden, Werte manuell eingeben"
    st.success(msg)

    if is_synth and n_dw > 0:
        is_manual_only = (
            n_dw == 1
            and dwellings_state[0].get("label", "").startswith("Manuelle Eingabe")
        )
        if is_manual_only:
            st.info(
                "ℹ️ Die GWR-API liefert für diese EGID weder Wohnungs-Detaildaten "
                "noch eine Wohnungs-Gesamtzahl. Du bekommst einen Standard-Eintrag "
                "mit Default-Werten (75 m² / 3 Zimmer). Trage die korrekten Werte "
                "der Wohnung unten manuell ein. Die Vorhersage funktioniert "
                "trotzdem, weil sie auf der Adresse und den Geo-Daten basiert."
            )
        else:
            st.info(
                f"ℹ️ Die GWR-Wohnungs-API liefert für diese EGID keine Detaildaten. "
                f"Wir zeigen **{n_dw} Standard-Einträge** mit Durchschnittsfläche aus "
                f"`garea / ganzwhg`. Du kannst Wohnfläche, Zimmer und Stockwerk pro "
                f"Wohnung manuell anpassen."
            )

    # ---------- Wohnungs-Dropdown (nur wenn welche da sind) ----------
    if n_dw > 0:
        def _fmt_dwelling(idx: int) -> str:
            d = dwellings_state[idx]
            ew    = d.get("ewid") or "?"
            label = d.get("label") or ""
            floor = d.get("floor_label") or "?"
            r     = d.get("rooms")
            a     = d.get("area")
            r_str = f"{r:.1f}".rstrip("0").rstrip(".") if r else "?"
            a_str = f"{a:.0f}" if a else "?"
            label_part = f" · {label}" if label else ""
            return f"EWID {ew}{label_part}  ·  {floor}  ·  {r_str} Zi  ·  {a_str} m²"

        # idx in Range halten (falls Dwellings sich geändert haben)
        if not (0 <= st.session_state.dwelling_selector_idx < n_dw):
            st.session_state.dwelling_selector_idx = 0

        st.selectbox(
            "🏠 Wohnung auswählen",
            range(n_dw),
            format_func=_fmt_dwelling,
            key="dwelling_selector_idx",
            on_change=_on_dwelling_change,
        )

    # ---------- Auto-gefüllte Inputs (mit Override) ----------
    # Standard-Floor-Optionen + alle vorkommenden Floors aus den Dwellings
    base_floor_options = ["—", "EG",
                           "1. OG", "2. OG", "3. OG", "4. OG",
                           "5. OG", "6. OG", "7. OG", "8. OG",
                           "1. UG", "2. UG", "UG", "1. DG", "2. DG"]
    floors_from_dwellings = [d.get("floor_label") for d in dwellings_state
                              if d.get("floor_label")]
    floor_options = list(dict.fromkeys(base_floor_options + floors_from_dwellings))
    # Aktuellen Wert sicher in den Optionen halten
    if st.session_state.lookup_floor not in floor_options:
        floor_options.append(st.session_state.lookup_floor)

    in_col1, in_col2, in_col3 = st.columns(3)
    in_col1.number_input(
        "Wohnfläche (m²)",
        min_value=10, max_value=500, step=1,
        help="Aus GWR übernommen — anpassbar.",
        key="lookup_area",  # session_state-bound, Wert kommt aus session_state
    )
    in_col2.number_input(
        "Zimmer",
        min_value=0.5, max_value=15.0, step=0.5,
        help="Aus GWR übernommen — anpassbar.",
        key="lookup_rooms",
    )
    in_col3.selectbox(
        "Stockwerk",
        floor_options,
        key="lookup_floor",
        help="Aktuell kein Modellfeature, nur informativ.",
    )

    # Lokale Variablen aus dem Session State (für Predict-Logik unten)
    area_lookup  = st.session_state.lookup_area
    rooms_lookup = st.session_state.lookup_rooms
    floor_input  = st.session_state.lookup_floor

    # ---------- Predict-Button ----------
    if st.button("🚀 Analysieren & Preis schätzen", type="primary",
                 key="lookup_predict_btn"):
        try:
            with st.spinner("Lade GWR-Gebäudedaten (Baujahr, Anzahl Whg., Grundstück) ..."):
                try:
                    gwr_info = load_gwr_data(
                        egid=egid_info_state.get("egid"),
                        gwr_link=egid_info_state.get("gwr_link"),
                    )
                except ValueError as gwr_err:
                    gwr_info = {"egid": egid_info_state.get("egid"),
                                "gbauj": None, "ganzwhg": None, "garea": None}
                    st.warning(f"⚠️ GWR-Daten unvollständig — {gwr_err}")

            with st.spinner("Lade Swisstopo (Höhe / ÖV / Solar / Bevölkerung) ..."):
                swisstopo_info = load_swisstopo_data(
                    east=egid_info_state["lv95_east"],
                    north=egid_info_state["lv95_north"],
                )

            raw_df = assemble_features(
                address        = egid_info_state["address"],
                area_sqm       = area_lookup,
                rooms          = rooms_lookup,
                floor          = None if floor_input == "—" else floor_input,
                egid_info      = egid_info_state,
                gwr_info       = gwr_info or {},
                swisstopo_info = swisstopo_info or {},
            )

            model_df, lookup_status = clean_and_finalize_records(raw_df)

            # Predict — die bestehende Funktion wird nur aufgerufen.
            try:
                pred_lookup = float(predictor.predict(model_df)[0])
            except Exception as e:
                pred_lookup = None
                st.error(f"Prediction fehlgeschlagen: {type(e).__name__}: {e}")

            # ----- Result-Anzeige -----
            if pred_lookup is not None:
                r1, r2, r3, r4 = st.columns(4)
                r1.metric(
                    "💰 Geschätzte Kaltmiete",
                    f"{pred_lookup:,.0f} CHF".replace(",", "'"),
                )
                r2.metric("Pro m²", f"{pred_lookup / area_lookup:.0f} CHF/m²")
                r3.metric("EGID", str(egid_info_state.get("egid") or "—"))
                r4.metric(
                    "Stockwerk",
                    "—" if floor_input == "—" else floor_input,
                )

            for w in lookup_status.get("warnings", []):
                st.warning(w)

            with st.expander("🧾 API-Rohdaten (lookup_egid + load_gwr_data + load_swisstopo_data)"):
                api_rows = [
                    {"Quelle": "lookup_egid",
                     **{k: v for k, v in egid_info_state.items() if k != "feature_id"}},
                    {"Quelle": "load_gwr_data",
                     **{k: v for k, v in (gwr_info or {}).items() if k != "_attrs"}},
                    {"Quelle": "load_swisstopo_data",
                     **(swisstopo_info or {})},
                ]
                st.dataframe(
                    pd.DataFrame(api_rows).T.rename(columns={
                        0: "lookup_egid",
                        1: "load_gwr_data",
                        2: "load_swisstopo_data",
                    }),
                    use_container_width=True,
                )

            with st.expander("🔧 Bereinigte Modell-Eingabe (clean_and_finalize_records)"):
                st.dataframe(
                    model_df.T.rename(columns={0: "Wert"}),
                    use_container_width=True,
                )

        except requests.exceptions.RequestException as e:
            st.error(f"⏱️ API-Timeout / Verbindungsfehler: {e}")
        except Exception as e:
            st.error(f"Unerwarteter Fehler: {type(e).__name__}: {e}")

    # Optional: alle Wohnungen des Gebäudes als Liste anzeigen
    if n_dw > 0:
        with st.expander(f"🏘️ Alle {n_dw} Wohnungen im Gebäude"):
            dw_df = pd.DataFrame(dwellings_state)
            display_cols = [c for c in
                            ["ewid", "label", "floor_label", "rooms", "area", "year_built_dwelling"]
                            if c in dw_df.columns]
            st.dataframe(
                dw_df[display_cols].rename(columns={
                    "ewid":        "EWID",
                    "label":       "Bezeichnung",
                    "floor_label": "Stockwerk",
                    "rooms":       "Zimmer",
                    "area":        "Fläche (m²)",
                    "year_built_dwelling": "Wohnungs-Baujahr",
                }),
                use_container_width=True,
            )

    # Debug-Output: zeigen, falls 0 Wohnungen gefunden wurden — hilft beim
    # Diagnostizieren, was die API tatsächlich liefert.
    debug_info = st.session_state.get("lookup_dwellings_debug") or {}
    if debug_info and n_dw == 0:
        with st.expander("🐛 Debug: was hat die GWR-API geliefert?", expanded=False):
            st.write(
                "Wenn hier `top_keys` einen Wohnungs-bezogenen Schlüssel zeigt, "
                "der noch nicht im Parser steht, gib mir Bescheid — "
                "dann kann ich die Heuristik gezielt erweitern."
            )
            st.json(debug_info)

st.divider()

# ===========================================================================
# 🎛️ Manuelle Eingabe (bestehender Flow — Sidebar)
# ===========================================================================
st.subheader("🎛️ Manuelle Eingabe (Sidebar-Sliders)")
st.caption("Alle Werte links in der Sidebar einstellen — Result direkt darunter.")

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
