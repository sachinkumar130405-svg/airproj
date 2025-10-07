# ml/ml_api.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
# near top of ml_api.py
import os
from fastapi import Header
# add near top of file
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
# Pydantic model
from pydantic import BaseModel
class TextPredictRequest(BaseModel):
    location_text: str
    ts: Optional[str] = None
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    temperature_2m: Optional[float] = None
    relativehumidity_2m: Optional[float] = None
    windspeed_10m: Optional[float] = None
    winddirection_10m: Optional[float] = None
    fire_count: float = 0.0
    mean_frp: float = 0.0

geolocator = Nominatim(user_agent="airproj_demo")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)

geolocator = Nominatim(user_agent="airproj_demo")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)


MODEL_PATH = r"C:\Users\SACHIN KUMAR\Desktop\airproj\ml\aqi_model_final.pkl"


RELOAD_TOKEN = os.environ.get("RELOAD_TOKEN", "dev-token-change-me")

def load_model():
    global model, feature_list
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    feature_list = model_data["meta"]["feature_list"]
    print("Model reloaded at", datetime.utcnow().isoformat())

ROOT = Path(__file__).resolve().parents[1]  # project root (one level above ml/)
PROCESSED_DIR = ROOT / "processed"
MERGED_PATH = PROCESSED_DIR / "merged" / "merged_dataset.parquet"
INTERP_PATH = PROCESSED_DIR / "merged" / "interpolated_grid.parquet"

# candidate model files (search order)
CANDIDATE_MODELS = [
    Path(__file__).resolve().parent / "aqi_model_final.pkl",
    Path(__file__).resolve().parent / "aqi_forecast_from_merged.pkl",
    Path(__file__).resolve().parent / "aqi_model.pkl",
    Path(__file__).resolve().parent / "model.pkl",
]


app = FastAPI(title="AQI Forecast + Spatial API", version="1.0")

# single CORSMiddleware entry (allow frontend on 5500 plus localhost for dev)
# allow all origins while developing (safe for local dev only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Utilities
# -------------------------
def interpret_aqi(aqi: float) -> str:
    aqi = float(aqi)
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def try_load_model(candidates):
    """Try to load first available joblib model file and return (model, meta, path) or (None,None,None)."""
    for p in candidates:
        try:
            if p.exists():
                data = joblib.load(p)
                # expected structure: {"model": ..., "meta": {"feature_list": [...]}, ...}
                model = data.get("model") if isinstance(data, dict) else data
                meta = data.get("meta") if isinstance(data, dict) else None
                return model, meta, p
        except Exception as e:
            # continue trying other candidates
            print(f"model load failed for {p}: {e}")
            continue
    return None, None, None


# -------------------------
# Load model (non-fatal)
# -------------------------
model, model_meta, model_path = try_load_model(CANDIDATE_MODELS)
feature_list = model_meta.get("feature_list") if model_meta else None

if model is not None:
    print(f"✅ Loaded model from {model_path} with {len(feature_list) if feature_list else 'unknown'} features.")
else:
    print("⚠️ No model found (service will still run but /predict will return 503).")
    print("Searched paths:", [str(p) for p in CANDIDATE_MODELS])


# -------------------------
# Input schema + feature builder
# -------------------------
class AQIInput(BaseModel):
    lat: float
    lon: float
    ts: str
    pm25: float
    pm10: float
    temperature_2m: float
    relativehumidity_2m: float
    windspeed_10m: float
    winddirection_10m: float
    fire_count: float = 0.0
    mean_frp: float = 0.0
    station_code: int = 0


def build_features(record: dict) -> pd.DataFrame:
    """Build a single-row features DataFrame matching model's feature_list.
    If feature_list is unknown, return a small default.
    """
    ts = pd.to_datetime(record["ts"], utc=True)
    df = pd.DataFrame([{
        **record,
        "hour": int(ts.hour),
        "dayofweek": int(ts.dayofweek),
        "month": int(ts.month),
        "weekofyear": int(ts.isocalendar().week),
        "temp_x_wind": record.get("temperature_2m", 0.0) * record.get("windspeed_10m", 0.0),
        "humidity_x_temp": record.get("relativehumidity_2m", 0.0) * record.get("temperature_2m", 0.0),
    }])
    # add lag/roll placeholders (the real system would fill these from historical data)
    for lag in [1, 3, 6, 24]:
        df[f"pm25_lag_{lag}"] = record.get("pm25", 0.0)
        df[f"pm10_lag_{lag}"] = record.get("pm10", 0.0)
    for roll in [3, 6, 24]:
        df[f"pm25_roll_{roll}"] = record.get("pm25", 0.0)
        df[f"pm10_roll_{roll}"] = record.get("pm10", 0.0)

    # If we have a model feature list, ensure all features exist and return in that order
    if feature_list:
        for f in feature_list:
            if f not in df.columns:
                df[f] = 0.0
        return df[feature_list]
    # fallback: return all columns
    return df


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "AQI Forecast + Spatial API running", "model_loaded": model is not None}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(model_path) if model_path else None,
        "num_features": len(feature_list) if feature_list else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


from fastapi import Request

from pydantic import BaseModel

class TextPredictRequest(BaseModel):
    location_text: str
    ts: str = None         # optional, default now if not provided
    pm25: float = None     # optional if you want to provide a measurement
    pm10: float = None
    temperature_2m: float = None
    relativehumidity_2m: float = None
    windspeed_10m: float = None
    winddirection_10m: float = None
    fire_count: float = 0.0
    mean_frp: float = 0.0

@app.post("/predict_text")
def predict_from_text(req: TextPredictRequest):
    ts = req.ts or datetime.utcnow().isoformat()
    loc = geocode(req.location_text)
    if loc is None:
        raise HTTPException(status_code=404, detail="Location not found")
    lat, lon = float(loc.latitude), float(loc.longitude)
    # Build payload and try to fill missing numeric features from MERGED_PATH
    payload = {
        "lat": lat, "lon": lon, "ts": ts,
        "pm25": req.pm25, "pm10": req.pm10,
        "temperature_2m": req.temperature_2m,
        "relativehumidity_2m": req.relativehumidity_2m,
        "windspeed_10m": req.windspeed_10m,
        "winddirection_10m": req.winddirection_10m,
        "fire_count": req.fire_count, "mean_frp": req.mean_frp,
        "station_code": 0
    }
    # best-effort fill from merged parquet
    try:
        if MERGED_PATH.exists():
            df = pd.read_parquet(MERGED_PATH)
            df = df.dropna(subset=["lat","lon"])
            coord_arr = np.vstack([df["lat"].to_numpy(), df["lon"].to_numpy()]).T
            d2 = np.sum((coord_arr - np.array([lat,lon]))**2, axis=1)
            idx = int(np.argmin(d2))
            row = df.iloc[idx]
            # map param/value if needed
            if payload["pm25"] is None:
                if "parameter" in row.index and row["parameter"] == "pm25" and "value" in row.index:
                    payload["pm25"] = float(row["value"])
                elif "pm25" in row.index:
                    payload["pm25"] = float(row["pm25"])
            for f in ["pm10","temperature_2m","relativehumidity_2m","windspeed_10m","winddirection_10m","fire_count","mean_frp"]:
                if payload.get(f) is None and f in row.index:
                    payload[f] = float(row[f])
    except Exception as e:
        print("predict_text fill warning:", e)

    # fallbacks so build_features won't crash
    for k in ["pm25","pm10","temperature_2m","relativehumidity_2m","windspeed_10m","winddirection_10m"]:
        if payload.get(k) is None:
            payload[k] = 0.0

    # build features and predict
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = build_features(payload)
        pred = model.predict(features)[0]
        pred = round(float(pred), 2)
        ts_next = pd.to_datetime(payload["ts"], utc=True) + timedelta(hours=1)
        return {
            "predicted_aqi": pred,
            "category": interpret_aqi(pred),
            "timestamp_predicted_for": ts_next.isoformat(),
            "location": {"lat": lat, "lon": lon, "address": loc.address}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import HTTPException

# Dummy station data (replace with live DB or CSV lookup later)
station_live_data = {
    101: {"temp": 25.6, "humidity": 42.3, "pm25": 38.7, "pm10": 70.5},
    102: {"temp": 29.1, "humidity": 50.2, "pm25": 52.9, "pm10": 85.1},
    103: {"temp": 21.3, "humidity": 35.8, "pm25": 18.4, "pm10": 40.2},
}

@app.get("/station-data/{station_code}")
async def get_station_data(station_code: int):
    if station_code not in station_live_data:
        raise HTTPException(status_code=404, detail="Station not found")

    return {
        "station_code": station_code,
        "timestamp": datetime.utcnow().isoformat(),
        "data": station_live_data[station_code]
    }


# -------------------------
# Interpolation (IDW) fallback
# -------------------------
def idw_interpolation(stations_df: pd.DataFrame, grid_res_deg: float = 0.01, power: float = 2.0, max_points: int = 5000) -> pd.DataFrame:
    if stations_df.shape[0] == 0:
        return pd.DataFrame(columns=["lat", "lon", "pm25"])
    min_lat, max_lat = float(stations_df["lat"].min()), float(stations_df["lat"].max())
    min_lon, max_lon = float(stations_df["lon"].min()), float(stations_df["lon"].max())
    # build grid arrays
    lats = np.arange(min_lat, max_lat + 1e-9, grid_res_deg)
    lons = np.arange(min_lon, max_lon + 1e-9, grid_res_deg)
    grid_pts = np.array(np.meshgrid(lats, lons)).T.reshape(-1, 2)
    pts = stations_df[["lat", "lon", "pm25"]].to_numpy()
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx, :]
    lat_s = pts[:, 0][:, None]
    lon_s = pts[:, 1][:, None]
    val_s = pts[:, 2][:, None]
    lat_g = grid_pts[:, 0][None, :]
    lon_g = grid_pts[:, 1][None, :]
    d2 = (lat_s - lat_g) ** 2 + (lon_s - lon_g) ** 2
    d = np.sqrt(d2)
    d[d == 0] = 1e-12
    w = 1.0 / (d ** power)
    num = (w * val_s).sum(axis=0)
    den = w.sum(axis=0)
    interp = num / den
    df_grid = pd.DataFrame({"lat": grid_pts[:, 0], "lon": grid_pts[:, 1], "pm25": interp})
    return df_grid


@app.get("/interpolate")
def interpolate(force: bool = Query(False), grid_res_deg: float = Query(0.01)):
    """Build spatial grid of PM2.5 using IDW from latest station values and save to INTERP_PATH."""
    try:
        INTERP_PATH.parent.mkdir(parents=True, exist_ok=True)
        if INTERP_PATH.exists() and not force:
            df_grid = pd.read_parquet(INTERP_PATH)
            return {"message": "Existing grid returned", "grid_points": len(df_grid), "path": str(INTERP_PATH)}
        if not MERGED_PATH.exists():
            raise HTTPException(status_code=404, detail=f"Merged dataset not found at {MERGED_PATH}")
        df = pd.read_parquet(MERGED_PATH)
        # take latest pm25 per (lat, lon)
        if "parameter" in df.columns:
            pm25_rows = df[df["parameter"] == "pm25"].sort_values("ts")
            if len(pm25_rows) > 0:
                latest = pm25_rows.groupby(["lat", "lon"], as_index=False).last().rename(columns={"value": "pm25"})
            else:
                latest = df.groupby(["lat", "lon"], as_index=False).last()
                if "value" in latest.columns:
                    latest = latest.rename(columns={"value": "pm25"})
        else:
            latest = df.groupby(["lat", "lon"], as_index=False).last()
            if "value" in latest.columns:
                latest = latest.rename(columns={"value": "pm25"})
        latest = latest.dropna(subset=["lat", "lon", "pm25"])
        latest["lat"] = latest["lat"].astype(float)
        latest["lon"] = latest["lon"].astype(float)
        latest["pm25"] = latest["pm25"].astype(float)
        df_grid = idw_interpolation(latest[["lat", "lon", "pm25"]], grid_res_deg=grid_res_deg)
        df_grid.to_parquet(INTERP_PATH, index=False)
        return {"message": "Grid computed and saved", "grid_points": len(df_grid), "path": str(INTERP_PATH)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Safe route (fallback)
# -------------------------
def sample_along_line(start, end, n_points=50):
    lats = np.linspace(start[0], end[0], n_points)
    lons = np.linspace(start[1], end[1], n_points)
    return list(zip(lats.tolist(), lons.tolist()))


def exposure_along_coords(coords, grid_df: pd.DataFrame):
    if grid_df is None or grid_df.empty:
        return {"avg_pm25": None, "sum_exposure": None, "pm_series": []}
    g_lat = grid_df["lat"].to_numpy()
    g_lon = grid_df["lon"].to_numpy()
    g_pm = grid_df["pm25"].to_numpy()
    pm_vals = []
    for (lat, lon) in coords:
        d2 = (g_lat - lat) ** 2 + (g_lon - lon) ** 2
        idx = int(np.argmin(d2))
        pm_vals.append(float(g_pm[idx]))
    pm_vals = np.array(pm_vals)
    avg = float(pm_vals.mean()) if pm_vals.size > 0 else None
    # compute segment lengths (km)
    seg_lengths = []
    for i in range(1, len(coords)):
        lat0, lon0 = coords[i - 1]
        lat1, lon1 = coords[i]
        dlat = np.radians(lat1 - lat0)
        dlon = np.radians(lon1 - lon0)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat0)) * np.cos(np.radians(lat1)) * np.sin(dlon / 2) ** 2
        R = 6371.0
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(max(0.0, 1 - a)))
        seg_lengths.append(R * c)
    seg_lengths = np.array(seg_lengths) if seg_lengths else np.array([0.0])
    pm_mid = (pm_vals[:-1] + pm_vals[1:]) / 2.0 if len(pm_vals) > 1 else pm_vals
    sum_exposure = float((pm_mid * seg_lengths).sum()) if len(pm_mid) > 0 else 0.0
    return {"avg_pm25": avg, "sum_exposure": sum_exposure, "pm_series": pm_vals.tolist()}

@app.get("/stations")
def list_stations():
    """Return list of known stations (lat, lon, name optional, last_aqi optional)."""
    try:
        if not MERGED_PATH.exists():
            raise HTTPException(status_code=404, detail=f"Merged dataset not found: {MERGED_PATH}")
        df = pd.read_parquet(MERGED_PATH)
        # prefer rows where parameter == pm25 to get last pm25; else use 'value'
        if "parameter" in df.columns:
            pm = df[df["parameter"] == "pm25"].copy()
            if pm.empty:
                pm = df.copy()
                pm["pm25"] = pm["value"]
            else:
                pm["pm25"] = pm["value"]
        else:
            pm = df.copy()
            if "value" in pm.columns:
                pm["pm25"] = pm["value"]
            else:
                pm["pm25"] = np.nan
        # get last reading per lat/lon
        pm = pm.sort_values("ts")
        station_last = pm.groupby(["lat", "lon"], as_index=False).last()[["lat", "lon", "pm25"]]
        # optional: create simple station IDs
        station_last["station_id"] = station_last.index + 1
        stations = station_last.to_dict(orient="records")
        return {"stations": stations, "count": len(stations)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/safe_route")
def get_safe_route(start_lat: float = Query(...), start_lon: float = Query(...),
                   end_lat: float = Query(...), end_lon: float = Query(...),
                   samples: int = Query(50)):
    try:
        if INTERP_PATH.exists():
            grid_df = pd.read_parquet(INTERP_PATH)
        else:
            # attempt to create it (best effort)
            _ = interpolate(force=False)
            grid_df = pd.read_parquet(INTERP_PATH) if INTERP_PATH.exists() else pd.DataFrame(columns=["lat", "lon", "pm25"])
        start = (float(start_lat), float(start_lon))
        end = (float(end_lat), float(end_lon))
        fast_coords = sample_along_line(start, end, n_points=5)
        safe_coords = sample_along_line(start, end, n_points=samples)
        exp_fast = exposure_along_coords(fast_coords, grid_df)
        exp_safe = exposure_along_coords(safe_coords, grid_df)
        return {
            "fast_route": {"coords": fast_coords, "exposure": exp_fast},
            "safe_route": {"coords": safe_coords, "exposure": exp_safe},
            "grid_points": len(grid_df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# initial load already done in your file
# add endpoint:
@app.post("/reload-model")
def reload_model(token: str = Header(...)):
    if token != RELOAD_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    try:
        load_model()
        return {"status": "ok", "reloaded_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from datetime import datetime, timedelta
import random

@app.get("/history/{station_code}")
async def get_station_history(station_code: int):
    """
    Generate synthetic AQI history for the past 24 hours.
    Returns time-series data for testing visualization.
    """
    now = datetime.utcnow()
    data = []
    aqi_base = random.randint(80, 180)

    for i in range(24):
        timestamp = now - timedelta(hours=i)
        aqi = aqi_base + random.uniform(-20, 20)
        data.append({
            "timestamp": timestamp.isoformat(),
            "aqi": round(aqi, 2),
            "category": (
                "Good" if aqi <= 50 else
                "Moderate" if aqi <= 100 else
                "Unhealthy" if aqi <= 200 else
                "Very Unhealthy"
            )
        })

    return {"station_code": station_code, "history": list(reversed(data))}

# --- Add to ml_api.py ---

from fastapi import Query
from typing import List, Dict

# Return latest station list for the map (one row per station)
@app.get("/stations")
def list_stations():
    """
    Returns latest reading for each station (grouped by lat/lon).
    Response:
    [{ "station_code": 101, "lat": 28.6139, "lon": 77.2090, "pm25": 72.1, "pm10": 120.2, "ts": "2025-10-07T..." }, ...]
    """
    if not MERGED_PATH.exists():
        raise HTTPException(status_code=404, detail="Merged dataset not found")
    df = pd.read_parquet(MERGED_PATH)
    # Ensure ts is datetime
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # Prefer pm25 rows; if parameter/value layout, pivot like earlier in pipeline
    if "parameter" in df.columns:
        pm25_df = df[df["parameter"] == "pm25"].copy()
        if pm25_df.empty:
            # fallback: pick last value grouping by lat/lon
            latest = df.groupby(["lat","lon"], as_index=False).last()
            latest = latest.rename(columns={"value":"pm25"})
        else:
            latest = pm25_df.groupby(["lat","lon"], as_index=False).last()
            latest = latest.rename(columns={"value":"pm25"})
    else:
        # already wide format (value column is pm25)
        latest = df.groupby(["lat","lon"], as_index=False).last()
        if "value" in latest.columns:
            latest = latest.rename(columns={"value":"pm25"})
    # Build result rows
    rows = []
    for _, r in latest.iterrows():
        row = {
            "station_code": int(r.get("station_code", 0)) if "station_code" in r else 0,
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "pm25": float(r["pm25"]) if pd.notna(r["pm25"]) else None,
            "ts": r.get("ts").isoformat() if "ts" in r and pd.notna(r["ts"]) else None
        }
        # include pm10 if available in wide df
        if "pm10" in r.index and pd.notna(r["pm10"]):
            row["pm10"] = float(r["pm10"])
        rows.append(row)
    return rows

# Return timeseries readings for a station (for popup chart)
@app.get("/readings")
def get_readings(station_code: int = Query(...), limit: int = Query(48)):
    """
    Returns last `limit` hourly readings for station_code.
    Response: [{ts:..., pm25:..., pm10:...}, ...]
    """
    if not MERGED_PATH.exists():
        raise HTTPException(status_code=404, detail="Merged dataset not found")
    df = pd.read_parquet(MERGED_PATH)
    # try match station_code column, otherwise lat/lon matching isn't done here
    if "station_code" in df.columns:
        s = df[df["station_code"] == station_code].sort_values("ts", ascending=False).head(limit)
    else:
        raise HTTPException(status_code=400, detail="station_code not available in dataset")
    # pivot if parameter/value
    if "parameter" in s.columns:
        s_pivot = s.pivot_table(index="ts", columns="parameter", values="value", aggfunc="last").reset_index()
    else:
        s_pivot = s.sort_values("ts", ascending=False)
    # format
    s_pivot["ts"] = pd.to_datetime(s_pivot["ts"], utc=True)
    out = []
    for _, row in s_pivot.iterrows():
        out.append({
            "ts": row["ts"].isoformat() if pd.notna(row["ts"]) else None,
            "pm25": float(row["pm25"]) if "pm25" in row.index and pd.notna(row["pm25"]) else None,
            "pm10": float(row["pm10"]) if "pm10" in row.index and pd.notna(row["pm10"]) else None
        })
    return out

# Optionally expose interpolation heat grid (if computed)
@app.get("/grid")
def get_grid():
    """
    Return interpolated grid (if exists) with fields: lat, lon, pm25
    """
    if not INTERP_PATH.exists():
        raise HTTPException(status_code=404, detail="Interpolated grid not found. Call /interpolate first.")
    g = pd.read_parquet(INTERP_PATH)
    return g.to_dict(orient="records")

