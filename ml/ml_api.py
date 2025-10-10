# ml/ml_api.py
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import os
import random
import math

# Optional libraries (osmnx + geopandas) — if not installed, code will still work with fallbacks
try:
    import osmnx as ox  # For real routes
    OSMNX_AVAILABLE = True
except Exception:
    OSMNX_AVAILABLE = False

try:
    import geopandas as gpd  # For geo handling (not strictly required in this file)
    GEOPANDAS_AVAILABLE = True
except Exception:
    GEOPANDAS_AVAILABLE = False

# Geopy for geocoding (used by /predict_text)
try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    geolocator = Nominatim(user_agent="airproj_demo")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False
    geocode = None

# --------------------
# Paths & model loader
# --------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "processed"
MERGED_PATH = PROCESSED_DIR / "merged" / "merged_dataset.parquet"
INTERP_PATH = PROCESSED_DIR / "merged" / "interpolated_grid.parquet"

CANDIDATE_MODELS = [
    Path(__file__).resolve().parent / "aqi_model_final.pkl",
    ROOT / "ml" / "aqi_model_final.pkl",
    ROOT / "aqi_model_final.pkl"
]

def try_load_model(candidates: List[Path]):
    for p in candidates:
        try:
            if p.exists():
                data = joblib.load(p)
                if isinstance(data, dict):
                    model = data.get("model")
                    meta = data.get("meta", None)
                else:
                    model = data
                    meta = None
                return model, meta, p
        except Exception as e:
            print(f"failed to load {p}: {e}")
            continue
    return None, None, None

model, model_meta, model_path = try_load_model(CANDIDATE_MODELS)
feature_list = model_meta.get("feature_list") if model_meta else None

# --------------------
# FastAPI app + CORS
# --------------------
app = FastAPI(title="AQI Forecast + Spatial API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Utilities
# --------------------
def interpret_aqi(aqi: float) -> str:
    aqi = float(aqi)
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

def haversine_meters(lat1, lon1, lat2, lon2):
    """Return distance in meters between two lat/lon points."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def build_features(record: dict) -> pd.DataFrame:
    """
    Build a one-row DataFrame from the provided record dictionary.
    If `feature_list` is defined (from model metadata), it will ensure columns match the model.
    """
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
        # Ensure all features exist in df (create with 0.0 if missing)
        for f in feature_list:
            if f not in df.columns:
                df[f] = 0.0
        # Return in same order as feature list
        return df[feature_list]
    return df

def idw_interpolation(data: pd.DataFrame, grid_res_deg=0.01, power=2):
    """
    Simple IDW interpolation for pm25 over a grid (latitude/longitude in degrees).
    Returns DataFrame with columns ['lat', 'lon', 'pm25'].
    """
    if data.empty:
        return pd.DataFrame(columns=["lat", "lon", "pm25"])
    # bounding box
    lat_min, lat_max = data["lat"].min() - grid_res_deg, data["lat"].max() + grid_res_deg
    lon_min, lon_max = data["lon"].min() - grid_res_deg, data["lon"].max() + grid_res_deg
    lat_vals = np.arange(lat_min, lat_max + grid_res_deg, grid_res_deg)
    lon_vals = np.arange(lon_min, lon_max + grid_res_deg, grid_res_deg)
    grid_lat, grid_lon = np.meshgrid(lat_vals, lon_vals)  # shapes: (len(lon_vals), len(lat_vals))
    pts = np.c_[grid_lat.ravel(), grid_lon.ravel()]  # each row is (lat, lon)

    # source points
    src_lat = data["lat"].values[:, None]  # shape (ns, 1)
    src_lon = data["lon"].values[:, None]  # shape (ns, 1)
    src_val = data["pm25"].values[:, None]  # shape (ns, 1)

    # grid points (1, ng)
    grid_lat_vec = pts[:, 0][None, :]  # (1, ng)
    grid_lon_vec = pts[:, 1][None, :]  # (1, ng)

    # distances (ns, ng)
    dlat = src_lat - grid_lat_vec
    dlon = src_lon - grid_lon_vec
    d2 = dlat**2 + dlon**2
    d = np.sqrt(d2)
    d[d == 0] = 1e-12  # avoid division by zero
    w = 1.0 / (d ** power)
    num = (w * src_val).sum(axis=0)
    den = w.sum(axis=0)
    interp = num / den
    df_grid = pd.DataFrame({
        "lat": pts[:, 0],
        "lon": pts[:, 1],
        "pm25": interp
    })
    return df_grid

# --------------------
# Pydantic models
# --------------------
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
    forecast_hours: int = 1  # Default 1, can be up to 24

class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    mode: Optional[str] = "drive"  # "drive", "walk", "bike" (used if OSMnx available)
    hours: int = 24

# --------------------
# Endpoints
# --------------------
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
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "osmnx_available": OSMNX_AVAILABLE,
        "geopy_available": GEOPY_AVAILABLE
    }

@app.post("/predict")
def predict(req: PredictInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    ts = pd.to_datetime(req.ts or datetime.utcnow().isoformat(), utc=True)
    payload = {
        "lat": req.lat,
        "lon": req.lon,
        "ts": ts.isoformat(),
        "pm25": req.pm25,
        "pm10": req.pm10,
        "temperature_2m": req.temperature_2m,
        "relativehumidity_2m": req.relativehumidity_2m,
        "windspeed_10m": req.windspeed_10m,
        "winddirection_10m": req.winddirection_10m,
        "fire_count": req.fire_count,
        "mean_frp": req.mean_frp,
        "station_code": req.station_code
    }
    # Fill pm values from MERGED_PATH if available and missing
    if MERGED_PATH.exists() and (payload["pm25"] is None or payload["pm10"] is None):
        try:
            df = pd.read_parquet(MERGED_PATH).dropna(subset=["lat","lon"])
            coords = np.vstack([df["lat"].to_numpy(), df["lon"].to_numpy()]).T
            d2 = np.sum((coords - np.array([req.lat, req.lon]))**2, axis=1)
            k = min(5, len(d2))
            idxs = np.argsort(d2)[:k]
            nearest = df.iloc[idxs]
            if payload["pm25"] is None and "pm25" in nearest:
                payload["pm25"] = float(nearest["pm25"].mean())
            if payload["pm10"] is None and "pm10" in nearest:
                payload["pm10"] = float(nearest["pm10"].mean())
        except Exception as e:
            print("failed to auto-fill pm values:", e)
    # fallback defaults
    if payload["pm25"] is None:
        payload["pm25"] = 50.0
    if payload["pm10"] is None:
        payload["pm10"] = 100.0

    features = build_features(payload)
    try:
        pred = model.predict(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    return {
        "location": {"lat": req.lat, "lon": req.lon},
        "ts": payload["ts"],
        "predicted_aqi": float(pred),
        "category": interpret_aqi(pred)
    }

@app.post("/predict_text")
def predict_from_text(req: TextPredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not GEOPY_AVAILABLE:
        raise HTTPException(status_code=500, detail="geopy not available on server")
    ts = pd.to_datetime(req.ts or datetime.utcnow().isoformat(), utc=True)
    loc = geocode(req.location_text)
    if loc is None:
        raise HTTPException(status_code=404, detail="Location not found")
    lat, lon = float(loc.latitude), float(loc.longitude)

    payload = {
        "lat": lat, "lon": lon, "ts": ts.isoformat(),
        "pm25": req.pm25, "pm10": req.pm10,
        "temperature_2m": req.temperature_2m,
        "relativehumidity_2m": req.relativehumidity_2m,
        "windspeed_10m": req.windspeed_10m,
        "winddirection_10m": req.winddirection_10m,
        "fire_count": req.fire_count, "mean_frp": req.mean_frp,
        "station_code": 0
    }

    # Fetch or estimate inputs from data if missing
    if MERGED_PATH.exists() and (payload["pm25"] is None or payload["pm10"] is None):
        try:
            df = pd.read_parquet(MERGED_PATH).dropna(subset=["lat","lon"])
            coords = np.vstack([df["lat"].to_numpy(), df["lon"].to_numpy()]).T
            d2 = np.sum((coords - np.array([lat,lon]))**2, axis=1)
            k = min(5, len(d2))
            idxs = np.argsort(d2)[:k]
            nearest = df.iloc[idxs]
            if payload["pm25"] is None and "pm25" in nearest:
                payload["pm25"] = float(nearest["pm25"].mean())
            if payload["pm10"] is None and "pm10" in nearest:
                payload["pm10"] = float(nearest["pm10"].mean())
        except Exception as e:
            print("failed to auto-fill pm values:", e)

    if payload["pm25"] is None:
        payload["pm25"] = 50.0
    if payload["pm10"] is None:
        payload["pm10"] = 100.0

    forecasts = []
    current_payload = payload.copy()
    max_hours = max(1, min(req.forecast_hours, 24))
    for h in range(max_hours):
        features = build_features(current_payload)
        try:
            pred = model.predict(features)[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
        forecasts.append({
            "hour": h + 1,
            "predicted_aqi": float(pred),
            "category": interpret_aqi(pred),
            "ts": (ts + timedelta(hours=h)).isoformat()
        })
        # Update payload for next hour (very simple simulation)
        current_payload["ts"] = forecasts[-1]["ts"]
        # set next pm25 to predicted aqi as a rough proxy (you may want to convert properly)
        current_payload["pm25"] = float(pred)
        # small perturbation for pm10 if present
        if current_payload.get("pm10") is not None:
            current_payload["pm10"] = float(current_payload["pm10"]) * 1.01

    return {"location": {"lat": lat, "lon": lon, "address": loc.address}, "forecasts": forecasts}

@app.get("/interpolate")
def interpolate(force: bool = Query(False), grid_res_deg: float = Query(0.01), power: float = Query(2.0)):
    """
    Compute or return an IDW interpolated pm25 grid from MERGED_PATH.
    If INTERP_PATH exists and force==False, return the cached interpolation.
    """
    if INTERP_PATH.exists() and not force:
        try:
            grid_df = pd.read_parquet(INTERP_PATH)
            return {"status": "ok", "cached": True, "n_points": len(grid_df), "sample": grid_df.head(5).to_dict(orient="records")}
        except Exception as e:
            print("failed to read cached interp:", e)
            # proceed to recompute

    if not MERGED_PATH.exists():
        raise HTTPException(status_code=404, detail="Merged dataset not found for interpolation")

    df = pd.read_parquet(MERGED_PATH).dropna(subset=["lat", "lon", "pm25"])
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available to interpolate")

    grid_df = idw_interpolation(df, grid_res_deg=grid_res_deg, power=power)
    # Ensure directory exists
    try:
        INTERP_PATH.parent.mkdir(parents=True, exist_ok=True)
        grid_df.to_parquet(INTERP_PATH, index=False)
    except Exception as e:
        print("warning: failed to save interpolated grid:", e)

    return {"status": "ok", "cached": False, "n_points": len(grid_df), "sample": grid_df.head(5).to_dict(orient="records")}

@app.post("/safe_route")
def get_safe_route(req: RouteRequest):
    """
    Returns route segments between start and end with sampled AQI (pm25) along the route.
    Uses OSMnx if available to compute realistic route; otherwise falls back to straight-line interpolation.
    """
    start = (req.start_lat, req.start_lon)
    end = (req.end_lat, req.end_lon)

    # Load interpolated grid if present
    grid_df = pd.DataFrame()
    if INTERP_PATH.exists():
        try:
            grid_df = pd.read_parquet(INTERP_PATH)
        except Exception as e:
            print("failed to load interpolated grid:", e)
            grid_df = pd.DataFrame()

    route_coords = []
    if OSMNX_AVAILABLE:
        try:
            # Use OSMnx to get route nodes and coordinates
            graph = ox.graph_from_point(start, dist=5000, network_type=req.mode)
            start_node = ox.nearest_nodes(graph, start[1], start[0])  # lon, lat
            end_node = ox.nearest_nodes(graph, end[1], end[0])
            route = ox.shortest_path(graph, start_node, end_node, weight="length")
            route_coords = [(graph.nodes[n]['y'], graph.nodes[n]['x']) for n in route]  # lat, lon
        except Exception as e:
            print("osmnx routing failed, falling back to straight line:", e)
            route_coords = [start, end]
    else:
        # fallback: straight line represented as start -> end
        route_coords = [start, end]

    # Sample AQI along route coords
    pm_vals = []
    for lat, lon in route_coords:
        if not grid_df.empty:
            # find nearest grid point
            d2 = (grid_df["lat"] - lat)**2 + (grid_df["lon"] - lon)**2
            idx = int(np.argmin(d2))
            pm_vals.append(float(grid_df.iloc[idx]["pm25"]))
        else:
            # fallback random
            pm_vals.append(float(random.uniform(50, 200)))

    # Build segments
    segments = []
    for i in range(1, len(route_coords)):
        seg_aqi = pm_vals[i-1]
        color = "green" if seg_aqi < 100 else "yellow" if seg_aqi < 200 else "red"
        segments.append({
            "start": {"lat": route_coords[i-1][0], "lon": route_coords[i-1][1]},
            "end": {"lat": route_coords[i][0], "lon": route_coords[i][1]},
            "aqi": seg_aqi,
            "color": color
        })

    # compute total distance as sum of haversine between consecutive coords
    total_distance_m = 0.0
    for i in range(1, len(route_coords)):
        total_distance_m += haversine_meters(route_coords[i-1][0], route_coords[i-1][1], route_coords[i][0], route_coords[i][1])

    avg_aqi = float(np.mean(pm_vals)) if pm_vals else 0.0
    return {"route_segments": segments, "avg_aqi": avg_aqi, "total_distance_m": total_distance_m}

# Reload model
RELOAD_TOKEN = os.environ.get("RELOAD_TOKEN", "dev-token-change-me")
@app.post("/reload-model")
def reload_model(token: str = Header(...)):
    if token != RELOAD_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    global model, model_meta, model_path, feature_list
    model, model_meta, model_path = try_load_model(CANDIDATE_MODELS)
    feature_list = model_meta.get("feature_list") if model_meta else None
    return {"status":"ok", "reloaded_at": datetime.utcnow().isoformat(), "model_loaded": model is not None}
