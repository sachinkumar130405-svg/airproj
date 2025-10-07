# ml/ml_api.py
from fastapi import FastAPI, HTTPException, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import os
import random

# Geocoding (optional, install geopy)
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    geolocator = Nominatim(user_agent="airproj_demo")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)
except Exception:
    geocode = None

# Paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "processed"
MERGED_PATH = PROCESSED_DIR / "merged" / "merged_dataset.parquet"
INTERP_PATH = PROCESSED_DIR / "merged" / "interpolated_grid.parquet"
MODEL_CANDIDATES = [
    Path(__file__).resolve().parent / "aqi_model_final.pkl",
    Path(__file__).resolve().parent / "aqi_model.pkl",
    Path(__file__).resolve().parent / "model.pkl",
]

MODEL_PATH = None
model = None
feature_list = None

def try_load_model(candidates: List[Path]):
    for p in candidates:
        try:
            if p.exists():
                data = joblib.load(p)
                if isinstance(data, dict) and "model" in data:
                    m = data["model"]
                    meta = data.get("meta", {})
                else:
                    m = data
                    meta = {}
                return m, meta.get("feature_list", None), p
        except Exception as e:
            print(f"Model load failed for {p}: {e}")
            continue
    return None, None, None

model, feature_list, MODEL_PATH = try_load_model(MODEL_CANDIDATES)
if model is not None:
    print(f"✅ Loaded model from {MODEL_PATH} with {len(feature_list) if feature_list else 'unknown'} features.")
else:
    print("⚠️ No model found. /predict will return 503 until a model is available.")

app = FastAPI(title="AQI Forecast + Spatial API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    ts = pd.to_datetime(record["ts"], utc=True)
    df = pd.DataFrame([{
        **record,
        "hour": int(ts.hour),
        "dayofweek": int(ts.dayofweek),
        "month": int(ts.month),
        "weekofyear": int(ts.isocalendar().week),
        "temp_x_wind": record.get("temperature_2m",0.0)*record.get("windspeed_10m",0.0),
        "humidity_x_temp": record.get("relativehumidity_2m",0.0)*record.get("temperature_2m",0.0),
    }])
    for lag in [1,3,6,24]:
        df[f"pm25_lag_{lag}"] = record.get("pm25", 0.0)
        df[f"pm10_lag_{lag}"] = record.get("pm10", 0.0)
    for roll in [3,6,24]:
        df[f"pm25_roll_{roll}"] = record.get("pm25", 0.0)
        df[f"pm10_roll_{roll}"] = record.get("pm10", 0.0)
    if feature_list:
        for f in feature_list:
            if f not in df.columns:
                df[f] = 0.0
        return df[feature_list]
    return df

@app.get("/")
def root():
    return {"message":"AQI Forecast + Spatial API running", "model_loaded": model is not None}

@app.get("/health")
def health():
    return {
        "status":"ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH else None,
        "num_features": len(feature_list) if feature_list else None,
        "timestamp": datetime.utcnow().isoformat()+"Z"
    }

@app.post("/predict")
def predict(input_data: AQIInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    rec = input_data.dict()
    try:
        features = build_features(rec)
        pred = model.predict(features)[0]
        pred = round(float(pred),2)
        ts_next = pd.to_datetime(rec["ts"], utc=True) + timedelta(hours=1)
        return {"predicted_aqi": pred, "category": interpret_aqi(pred), "timestamp_predicted_for": ts_next.isoformat(), "station_code": int(rec.get("station_code",0))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Natural language / place text -> predict
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

@app.post("/predict_text")
def predict_text(req: TextPredictRequest):
    if geocode is None:
        raise HTTPException(status_code=500, detail="Geocoding not available (install geopy).")
    ts = req.ts or datetime.utcnow().isoformat()
    loc = geocode(req.location_text)
    if loc is None:
        raise HTTPException(status_code=404, detail="Location not found")
    lat, lon = float(loc.latitude), float(loc.longitude)
    payload = {
        "lat": lat, "lon": lon, "ts": ts,
        "pm25": req.pm25, "pm10": req.pm10,
        "temperature_2m": req.temperature_2m,
        "relativehumidity_2m": req.relativehumidity_2m,
        "windspeed_10m": req.windspeed_10m,
        "winddirection_10m": req.winddirection_10m,
        "fire_count": req.fire_count, "mean_frp": req.mean_frp, "station_code": 0
    }
    # fill missing numeric from MERGED_PATH if present
    try:
        if MERGED_PATH.exists() and any(payload.get(k) is None for k in ["pm25","pm10","temperature_2m"]):
            df = pd.read_parquet(MERGED_PATH)
            df = df.dropna(subset=["lat","lon"])
            coords = np.vstack([df["lat"].to_numpy(), df["lon"].to_numpy()]).T
            d2 = np.sum((coords - np.array([lat,lon]))**2, axis=1)
            idx = int(np.argmin(d2))
            row = df.iloc[idx]
            for f in ["pm25","pm10","temperature_2m","relativehumidity_2m","windspeed_10m","winddirection_10m","fire_count","mean_frp"]:
                if payload.get(f) is None and f in row.index:
                    if f in ["pm25","pm10"] and "parameter" in row.index and row.get("parameter")==f and "value" in row.index:
                        payload[f] = float(row["value"])
                    else:
                        try:
                            payload[f] = float(row[f])
                        except Exception:
                            pass
    except Exception as e:
        print("Geocode-fill failed:", e)

    # ensure defaults
    for k in ["pm25","pm10","temperature_2m","relativehumidity_2m","windspeed_10m","winddirection_10m"]:
        if payload.get(k) is None:
            payload[k] = 0.0

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = build_features(payload)
        pred = model.predict(features)[0]
        pred = round(float(pred),2)
        ts_next = pd.to_datetime(payload["ts"], utc=True) + timedelta(hours=1)
        return {"predicted_aqi": pred, "category": interpret_aqi(pred), "timestamp_predicted_for": ts_next.isoformat(), "location": {"lat": lat, "lon": lon, "address": loc.address}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Interpolation (IDW)
def idw_interpolation(stations_df: pd.DataFrame, grid_res_deg: float = 0.01, power: float = 2.0):
    if stations_df.shape[0] == 0:
        return pd.DataFrame(columns=["lat","lon","pm25"])
    min_lat, max_lat = float(stations_df["lat"].min()), float(stations_df["lat"].max())
    min_lon, max_lon = float(stations_df["lon"].min()), float(stations_df["lon"].max())
    lats = np.arange(min_lat, max_lat + 1e-9, grid_res_deg)
    lons = np.arange(min_lon, max_lon + 1e-9, grid_res_deg)
    grid_pts = np.array(np.meshgrid(lats, lons)).T.reshape(-1,2)
    pts = stations_df[["lat","lon","pm25"]].to_numpy()
    lat_s = pts[:,0][:,None]
    lon_s = pts[:,1][:,None]
    val_s = pts[:,2][:,None]
    lat_g = grid_pts[:,0][None,:]
    lon_g = grid_pts[:,1][None,:]
    d2 = (lat_s - lat_g)**2 + (lon_s - lon_g)**2
    d = np.sqrt(d2)
    d[d==0] = 1e-12
    w = 1.0/(d**power)
    num = (w*val_s).sum(axis=0)
    den = w.sum(axis=0)
    interp = num/den
    return pd.DataFrame({"lat": grid_pts[:,0], "lon": grid_pts[:,1], "pm25": interp})

@app.get("/interpolate")
def interpolate(force: bool = Query(False), grid_res_deg: float = Query(0.01)):
    INTERP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if INTERP_PATH.exists() and not force:
        df_grid = pd.read_parquet(INTERP_PATH)
        return {"message":"Existing grid returned","grid_points":len(df_grid),"path":str(INTERP_PATH)}
    if not MERGED_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Merged dataset not found: {MERGED_PATH}")
    df = pd.read_parquet(MERGED_PATH)
    # prefer pm25 rows
    if "parameter" in df.columns:
        pm = df[df["parameter"]=="pm25"].copy()
        if pm.empty:
            latest = df.groupby(["lat","lon"], as_index=False).last()
            if "value" in latest.columns:
                latest = latest.rename(columns={"value":"pm25"})
        else:
            latest = pm.groupby(["lat","lon"], as_index=False).last().rename(columns={"value":"pm25"})
    else:
        latest = df.groupby(["lat","lon"], as_index=False).last()
        if "value" in latest.columns:
            latest = latest.rename(columns={"value":"pm25"})
    latest = latest.dropna(subset=["lat","lon","pm25"])
    latest["lat"] = latest["lat"].astype(float)
    latest["lon"] = latest["lon"].astype(float)
    latest["pm25"] = latest["pm25"].astype(float)
    grid_df = idw_interpolation(latest[["lat","lon","pm25"]], grid_res_deg=grid_res_deg)
    grid_df.to_parquet(INTERP_PATH, index=False)
    return {"message":"Grid computed and saved","grid_points":len(grid_df),"path":str(INTERP_PATH)}

@app.get("/grid")
def get_grid():
    if not INTERP_PATH.exists():
        raise HTTPException(status_code=404, detail="Interpolated grid not found. Call /interpolate first.")
    g = pd.read_parquet(INTERP_PATH)
    return g.to_dict(orient="records")

@app.get("/stations")
def list_stations():
    if not MERGED_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Merged dataset not found: {MERGED_PATH}")
    df = pd.read_parquet(MERGED_PATH)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if "parameter" in df.columns:
        pm = df[df["parameter"]=="pm25"].copy()
        if pm.empty:
            latest = df.groupby(["lat","lon"], as_index=False).last()
            if "value" in latest.columns:
                latest = latest.rename(columns={"value":"pm25"})
        else:
            latest = pm.groupby(["lat","lon"], as_index=False).last().rename(columns={"value":"pm25"})
    else:
        latest = df.groupby(["lat","lon"], as_index=False).last()
        if "value" in latest.columns:
            latest = latest.rename(columns={"value":"pm25"})
    rows = []
    for _, r in latest.iterrows():
        rows.append({
            "station_code": int(r.get("station_code", 0)) if "station_code" in r else 0,
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "pm25": float(r["pm25"]) if pd.notna(r["pm25"]) else None,
            "ts": r.get("ts").isoformat() if "ts" in r and pd.notna(r["ts"]) else None
        })
    return {"stations": rows, "count": len(rows)}

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
        d2 = (g_lat - lat)**2 + (g_lon - lon)**2
        idx = int(np.argmin(d2))
        pm_vals.append(float(g_pm[idx]))
    pm_vals = np.array(pm_vals)
    avg = float(pm_vals.mean()) if pm_vals.size>0 else None
    seg_lengths = []
    for i in range(1,len(coords)):
        lat0, lon0 = coords[i-1]
        lat1, lon1 = coords[i]
        dlat = np.radians(lat1-lat0)
        dlon = np.radians(lon1-lon0)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat0))*np.cos(np.radians(lat1))*np.sin(dlon/2)**2
        R = 6371.0
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(max(0.0,1-a)))
        seg_lengths.append(R*c)
    seg_lengths = np.array(seg_lengths) if seg_lengths else np.array([0.0])
    pm_mid = (pm_vals[:-1] + pm_vals[1:]) / 2.0 if len(pm_vals)>1 else pm_vals
    sum_exposure = float((pm_mid * seg_lengths).sum()) if len(pm_mid)>0 else 0.0
    return {"avg_pm25": avg, "sum_exposure": sum_exposure, "pm_series": pm_vals.tolist()}

@app.get("/safe_route")
def get_safe_route(start_lat: float = Query(...), start_lon: float = Query(...), end_lat: float = Query(...), end_lon: float = Query(...), samples: int = Query(50)):
    grid_df = pd.DataFrame(columns=["lat","lon","pm25"])
    if INTERP_PATH.exists():
        grid_df = pd.read_parquet(INTERP_PATH)
    else:
        try:
            _ = interpolate(force=False)
            if INTERP_PATH.exists():
                grid_df = pd.read_parquet(INTERP_PATH)
        except Exception:
            pass
    start = (float(start_lat), float(start_lon))
    end = (float(end_lat), float(end_lon))
    fast_coords = sample_along_line(start, end, n_points=5)
    safe_coords = sample_along_line(start, end, n_points=samples)
    exp_fast = exposure_along_coords(fast_coords, grid_df)
    exp_safe = exposure_along_coords(safe_coords, grid_df)
    return {"fast_route": {"coords": fast_coords, "exposure": exp_fast}, "safe_route": {"coords": safe_coords, "exposure": exp_safe}, "grid_points": len(grid_df)}

# reload-model endpoint (protected by a header token)
RELOAD_TOKEN = os.environ.get("RELOAD_TOKEN", "dev-token-change-me")
def load_model_endpoint():
    global model, feature_list, MODEL_PATH
    m, f, p = try_load_model(MODEL_CANDIDATES)
    model, feature_list, MODEL_PATH = m, f, p
    return model is not None

@app.post("/reload-model")
def reload_model(token: str = Header(...)):
    if token != RELOAD_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    ok = load_model_endpoint()
    if not ok:
        raise HTTPException(status_code=500, detail="Reload failed")
    return {"status":"ok", "model_path": str(MODEL_PATH)}

# small synthetic history for UI testing
@app.get("/history/{station_code}")
def history(station_code: int):
    now = datetime.utcnow()
    series = []
    base = random.randint(60,140)
    for i in range(24):
        t = now - timedelta(hours=i)
        a = base + random.uniform(-20,20)
        series.append({"ts": t.isoformat(), "aqi": round(a,2)})
    return {"station_code": station_code, "history": list(reversed(series))}
