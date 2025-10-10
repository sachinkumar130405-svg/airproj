# ml/ml_api.py
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import os
import random

# Geopy for NL geocoding
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent="airproj_demo")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)

# Paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "processed"
MERGED_PATH = PROCESSED_DIR / "merged" / "merged_dataset.parquet"
INTERP_PATH = PROCESSED_DIR / "merged" / "interpolated_grid.parquet"

# Candidate model path (look in ml/ first then root)
CANDIDATE_MODELS = [
    Path(__file__).resolve().parent / "aqi_model_final.pkl",
    ROOT / "ml" / "aqi_model_final.pkl",
    ROOT / "aqi_model_final.pkl"
]
MODEL_PATH = None

def try_load_model(candidates):
    for p in candidates:
        try:
            if p.exists():
                data = joblib.load(p)
                model = data.get("model") if isinstance(data, dict) else data
                meta = data.get("meta") if isinstance(data, dict) else None
                return model, meta, p
        except Exception as e:
            print(f"failed to load {p}: {e}")
            continue
    return None, None, None

model, model_meta, model_path = try_load_model(CANDIDATE_MODELS)
feature_list = model_meta.get("feature_list") if model_meta else None

app = FastAPI(title="AQI Forecast + Spatial API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

def interpret_aqi(aqi: float) -> str:
    aqi = float(aqi)
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

def build_features(record: dict) -> pd.DataFrame:
    ts = pd.to_datetime(record["ts"], utc=True)
    df = pd.DataFrame([{
        **record,
        "hour": int(ts.hour),
        "dayofweek": int(ts.dayofweek),
        "month": int(ts.month),
    }])
    # lag placeholders
    for lag in [1,3,6,24]:
        df[f"pm25_lag_{lag}"] = record.get("pm25", 0.0)
        df[f"pm10_lag_{lag}"] = record.get("pm10", 0.0)
    # rolling placeholders
    for roll in [3,6,24]:
        df[f"pm25_roll_{roll}"] = record.get("pm25", 0.0)
        df[f"pm10_roll_{roll}"] = record.get("pm10", 0.0)
    if feature_list:
        for f in feature_list:
            if f not in df.columns:
                df[f] = 0.0
        return df[feature_list]
    return df

class PredictInput(BaseModel):
    lat: float
    lon: float
    ts: Optional[str] = None
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    temperature_2m: Optional[float] = None
    relativehumidity_2m: Optional[float] = None
    windspeed_10m: Optional[float] = None
    winddirection_10m: Optional[float] = None
    fire_count: float = 0.0
    mean_frp: float = 0.0
    station_code: int = 0

@app.get("/")
def root():
    return {"message":"AQI Forecast + Spatial API running", "model_loaded": model is not None}

@app.get("/health")
def health_check():
    return {
        "status":"ok",
        "model_loaded": model is not None,
        "model_path": str(model_path) if model_path else None,
        "num_features": len(feature_list) if feature_list else None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# Text -> predict endpoint
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
def predict_from_text(req: TextPredictRequest):
    ts = req.ts or datetime.utcnow().isoformat()

    # geocode
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
        "fire_count": req.fire_count, "mean_frp": req.mean_frp,
        "station_code": 0
    }

    # If some core inputs are missing, try to estimate from nearby points:
    try:
        if MERGED_PATH.exists() and (payload["pm25"] is None or payload["pm10"] is None):
            df = pd.read_parquet(MERGED_PATH).dropna(subset=["lat","lon"])
            coords = np.vstack([df["lat"].to_numpy(), df["lon"].to_numpy()]).T
            # compute squared distance
            d2 = np.sum((coords - np.array([lat,lon]))**2, axis=1)
            # choose k nearest up to 5
            k = min(5, len(d2))
            idxs = np.argsort(d2)[:k]
            weights = 1.0 / (np.sqrt(d2[idxs]) + 1e-6)
            weights = weights / weights.sum()
            # try to extract pm25/pm10/meteorology from these rows (if 'value'/'parameter' layout, pivot first)
            block = df.iloc[idxs]
            # if df has parameter/value layout, pick rows where parameter==...
            def weighted_field(field_name, param_name=None):
                vals = []
                for _, r in block.iterrows():
                    if param_name and "parameter" in r.index and r["parameter"] != param_name:
                        vals.append(np.nan)
                    elif field_name in r.index and pd.notna(r[field_name]):
                        vals.append(float(r[field_name]) if field_name!='value' else float(r['value']))
                    elif "value" in r.index and param_name and r.get("parameter")==param_name:
                        vals.append(float(r["value"]))
                    else:
                        vals.append(np.nan)
                vals = np.array(vals, dtype=float)
                mask = ~np.isnan(vals)
                if mask.sum()==0:
                    return None
                w = weights.copy()
                w[~mask] = 0.0
                if w.sum() == 0:
                    return None
                return float((w * np.nan_to_num(vals)).sum() / (w.sum()))
            # fill pm25/pm10 first
            if payload["pm25"] is None:
                v = weighted_field("value", param_name="pm25") if "parameter" in block.columns else weighted_field("pm25")
                if v is not None: payload["pm25"] = v
            if payload["pm10"] is None:
                v = weighted_field("value", param_name="pm10") if "parameter" in block.columns else weighted_field("pm10")
                if v is not None: payload["pm10"] = v
            # fill simple meteorology if missing
            for f in ["temperature_2m","relativehumidity_2m","windspeed_10m","winddirection_10m","fire_count","mean_frp"]:
                if payload.get(f) is None:
                    v = weighted_field(f)
                    if v is not None:
                        payload[f] = v
    except Exception as e:
        print("geocode-fill failed:", e)

    # final defaults so build_features doesn't crash
    for k in ["pm25","pm10","temperature_2m","relativehumidity_2m","windspeed_10m","winddirection_10m"]:
        if payload.get(k) is None:
            payload[k] = 0.0

    # predict
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = build_features(payload)
        pred = model.predict(features)[0]
        pred = round(float(pred),2)
        ts_next = pd.to_datetime(payload["ts"], utc=True) + timedelta(hours=1)
        return {
            "predicted_aqi": pred,
            "category": interpret_aqi(pred),
            "timestamp_predicted_for": ts_next.isoformat(),
            "location": {"lat": lat, "lon": lon, "address": loc.address}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# POST /predict (single record) - reuse same build_features
@app.post("/predict")
def predict_record(inp: PredictInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    payload = inp.dict()
    payload["ts"] = payload.get("ts") or datetime.utcnow().isoformat()
    for k in ["pm25","pm10","temperature_2m","relativehumidity_2m","windspeed_10m","winddirection_10m"]:
        if payload.get(k) is None:
            payload[k] = 0.0
    try:
        features = build_features(payload)
        pred = model.predict(features)[0]
        pred = round(float(pred),2)
        ts_next = pd.to_datetime(payload["ts"], utc=True) + timedelta(hours=1)
        return {"predicted_aqi": pred, "category": interpret_aqi(pred), "timestamp_predicted_for": ts_next.isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Stations endpoint - return rows array
@app.get("/stations")
def list_stations():
    if not MERGED_PATH.exists():
        raise HTTPException(status_code=404, detail="Merged dataset not found")
    df = pd.read_parquet(MERGED_PATH)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # prefer pm25 parameter if present
    if "parameter" in df.columns:
        pm = df[df["parameter"] == "pm25"].copy()
        if pm.empty:
            latest = df.groupby(["lat","lon"], as_index=False).last()
            if "value" in latest.columns: latest = latest.rename(columns={"value":"pm25"})
        else:
            latest = pm.groupby(["lat","lon"], as_index=False).last().rename(columns={"value":"pm25"})
    else:
        latest = df.groupby(["lat","lon"], as_index=False).last()
        if "value" in latest.columns: latest = latest.rename(columns={"value":"pm25"})
    rows = []
    for _, r in latest.iterrows():
        rows.append({
            "station_code": int(r.get("station_code", 0)) if "station_code" in r.index else 0,
            "lat": float(r["lat"]), "lon": float(r["lon"]),
            "pm25": float(r["pm25"]) if pd.notna(r["pm25"]) else None,
            "ts": r.get("ts").isoformat() if "ts" in r.index and pd.notna(r["ts"]) else None
        })
    return {"stations": rows, "count": len(rows)}

@app.get("/readings")
def get_readings(station_code: int = Query(...), limit: int = Query(48)):
    if not MERGED_PATH.exists():
        raise HTTPException(status_code=404, detail="Merged dataset not found")
    df = pd.read_parquet(MERGED_PATH)
    if "station_code" not in df.columns:
        raise HTTPException(status_code=400, detail="station_code not available in dataset")
    s = df[df["station_code"] == station_code].sort_values("ts", ascending=False).head(limit)
    if "parameter" in s.columns:
        s_pivot = s.pivot_table(index="ts", columns="parameter", values="value", aggfunc="last").reset_index()
    else:
        s_pivot = s.sort_values("ts", ascending=False)
    s_pivot["ts"] = pd.to_datetime(s_pivot["ts"], utc=True)
    out = []
    for _, row in s_pivot.iterrows():
        out.append({"ts": row["ts"].isoformat() if pd.notna(row["ts"]) else None,
                    "pm25": float(row["pm25"]) if "pm25" in row.index and pd.notna(row["pm25"]) else None,
                    "pm10": float(row["pm10"]) if "pm10" in row.index and pd.notna(row["pm10"]) else None})
    return out

@app.get("/history/{station_code}")
def get_station_history(station_code: int):
    # synthetic fallback if DB not available
    now = datetime.utcnow()
    data = []
    aqi_base = random.randint(60,150)
    for i in range(24):
        timestamp = now - timedelta(hours=(23-i))
        aqi = aqi_base + random.uniform(-20,20)
        data.append({"timestamp": timestamp.isoformat() + "Z", "aqi": round(aqi,2),
                     "category": interpret_aqi(aqi)})
    return {"station_code": station_code, "history": data}

# IDW interpolation (server-side)
def idw_interpolation(stations_df: pd.DataFrame, grid_res_deg: float = 0.01, power: float = 2.0, max_points: int = 5000):
    if stations_df.shape[0] == 0:
        return pd.DataFrame(columns=["lat","lon","pm25"])
    min_lat, max_lat = float(stations_df["lat"].min()), float(stations_df["lat"].max())
    min_lon, max_lon = float(stations_df["lon"].min()), float(stations_df["lon"].max())
    lats = np.arange(min_lat, max_lat + 1e-9, grid_res_deg)
    lons = np.arange(min_lon, max_lon + 1e-9, grid_res_deg)
    grid_pts = np.array(np.meshgrid(lats, lons)).T.reshape(-1,2)
    pts = stations_df[["lat","lon","pm25"]].to_numpy()
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx,:]
    lat_s = pts[:,0][:,None]; lon_s = pts[:,1][:,None]; val_s = pts[:,2][:,None]
    lat_g = grid_pts[:,0][None,:]; lon_g = grid_pts[:,1][None,:]
    d2 = (lat_s - lat_g)**2 + (lon_s - lon_g)**2
    d = np.sqrt(d2); d[d==0] = 1e-12
    w = 1.0 / (d ** power)
    num = (w * val_s).sum(axis=0); den = w.sum(axis=0)
    interp = num / den
    df_grid = pd.DataFrame({"lat": grid_pts[:,0], "lon": grid_pts[:,1], "pm25": interp})
    return df_grid

@app.get("/interpolate")
def interpolate(force: bool = Query(False), grid_res_deg: float = Query(0.01)):
    try:
        INTERP_PATH.parent.mkdir(parents=True, exist_ok=True)
        if INTERP_PATH.exists() and not force:
            df_grid = pd.read_parquet(INTERP_PATH)
            return {"message": "Existing grid returned", "grid_points": len(df_grid), "path": str(INTERP_PATH)}
        if not MERGED_PATH.exists():
            raise HTTPException(status_code=404, detail=f"Merged dataset not found at {MERGED_PATH}")
        df = pd.read_parquet(MERGED_PATH)
        if "parameter" in df.columns:
            pm25_rows = df[df["parameter"] == "pm25"].sort_values("ts")
            if len(pm25_rows) > 0:
                latest = pm25_rows.groupby(["lat","lon"], as_index=False).last().rename(columns={"value":"pm25"})
            else:
                latest = df.groupby(["lat","lon"], as_index=False).last()
                if "value" in latest.columns:
                    latest = latest.rename(columns={"value":"pm25"})
        else:
            latest = df.groupby(["lat","lon"], as_index=False).last()
            if "value" in latest.columns:
                latest = latest.rename(columns={"value":"pm25"})
        latest = latest.dropna(subset=["lat","lon","pm25"])
        latest["lat"] = latest["lat"].astype(float); latest["lon"] = latest["lon"].astype(float); latest["pm25"] = latest["pm25"].astype(float)
        df_grid = idw_interpolation(latest[["lat","lon","pm25"]], grid_res_deg=grid_res_deg)
        df_grid.to_parquet(INTERP_PATH, index=False)
        return {"message":"Grid computed and saved", "grid_points": len(df_grid), "path": str(INTERP_PATH)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grid")
def get_grid():
    if not INTERP_PATH.exists():
        raise HTTPException(status_code=404, detail="Interpolated grid not found. Call /interpolate first.")
    g = pd.read_parquet(INTERP_PATH)
    return g.to_dict(orient="records")

# sample along line + exposure
def sample_along_line(start, end, n_points=50):
    lats = np.linspace(start[0], end[0], n_points)
    lons = np.linspace(start[1], end[1], n_points)
    return list(zip(lats.tolist(), lons.tolist()))

def exposure_along_coords(coords, grid_df: pd.DataFrame):
    if grid_df is None or grid_df.empty:
        return {"avg_pm25": None, "sum_exposure": None, "pm_series": []}
    g_lat = grid_df["lat"].to_numpy(); g_lon = grid_df["lon"].to_numpy(); g_pm = grid_df["pm25"].to_numpy()
    pm_vals = []
    for (lat, lon) in coords:
        d2 = (g_lat - lat)**2 + (g_lon - lon)**2
        idx = int(np.argmin(d2))
        pm_vals.append(float(g_pm[idx]))
    pm_vals = np.array(pm_vals)
    avg = float(pm_vals.mean()) if pm_vals.size>0 else None
    seg_lengths = []
    for i in range(1,len(coords)):
        lat0, lon0 = coords[i-1]; lat1, lon1 = coords[i]
        dlat = np.radians(lat1-lat0); dlon = np.radians(lon1-lon0)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat0))*np.cos(np.radians(lat1))*np.sin(dlon/2)**2
        R = 6371.0; c = 2*np.arctan2(np.sqrt(a), np.sqrt(max(0.0,1-a)))
        seg_lengths.append(R*c)
    seg_lengths = np.array(seg_lengths) if seg_lengths else np.array([0.0])
    pm_mid = (pm_vals[:-1] + pm_vals[1:]) / 2.0 if len(pm_vals) > 1 else pm_vals
    sum_exposure = float((pm_mid * seg_lengths).sum()) if len(pm_mid)>0 else 0.0
    return {"avg_pm25": avg, "sum_exposure": sum_exposure, "pm_series": pm_vals.tolist()}

@app.get("/safe_route")
def get_safe_route(start_lat: float = Query(...), start_lon: float = Query(...),
                   end_lat: float = Query(...), end_lon: float = Query(...),
                   samples: int = Query(50)):
    try:
        if INTERP_PATH.exists():
            grid_df = pd.read_parquet(INTERP_PATH)
        else:
            _ = interpolate(force=False)
            grid_df = pd.read_parquet(INTERP_PATH) if INTERP_PATH.exists() else pd.DataFrame(columns=["lat","lon","pm25"])
        start = (float(start_lat), float(start_lon)); end = (float(end_lat), float(end_lon))
        fast_coords = sample_along_line(start, end, n_points=5)
        safe_coords = sample_along_line(start, end, n_points=samples)
        exp_fast = exposure_along_coords(fast_coords, grid_df)
        exp_safe = exposure_along_coords(safe_coords, grid_df)
        return {"fast_route": {"coords": fast_coords, "exposure": exp_fast}, "safe_route": {"coords": safe_coords, "exposure": exp_safe}, "grid_points": len(grid_df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# reload model endpoint (header token)
RELOAD_TOKEN = os.environ.get("RELOAD_TOKEN", "dev-token-change-me")
@app.post("/reload-model")
def reload_model(token: str = Header(...)):
    if token != RELOAD_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    try:
        global model, model_meta, model_path, feature_list
        model, model_meta, model_path = try_load_model(CANDIDATE_MODELS)
        feature_list = model_meta.get("feature_list") if model_meta else None
        return {"status":"ok", "reloaded_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
