# merge_datasets.py
"""
Phase 2: Merge OpenAQ readings + FIRMS hotspots + Meteo into a single cleaned Parquet.

Outputs:
- processed\merged\merged_dataset.parquet
- processed\merged\station_hotspot_counts.parquet

Usage:
& "C:/.../python.exe" "C:/.../etl/merge_datasets.py"
"""

import os
import glob
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin

# ------------------ CONFIG: update only if your folders differ ------------------
PROCESSED_OPENAQ_DIR = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\OPEN AQ"
PROCESSED_FIRMS_DIR = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\FIRMS VIIRS"
PROCESSED_METEO_DIR = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\METEO"
OUT_DIR = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\merged"
FIRMS_FILENAME_GLOB = os.path.join(PROCESSED_FIRMS_DIR, "*.parquet")
OPENAQ_FILENAME_GLOB = "processed/open_aq/*.parquet"

METEO_FILENAME_GLOB = os.path.join(PROCESSED_METEO_DIR, "*.parquet")
OPENAQ_GLOB = os.path.join(PROCESSED_OPENAQ_DIR, "*.parquet")
# hotspot radii (km) to compute counts for
RADII_KM = [10.0, 50.0]
# floor readings to hourly for merging with meteo/hotspots
TIME_FLOOR = "H"
# ---------------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

pd.options.mode.chained_assignment = None  # silence SettingWithCopyWarning for clarity

def haversine_km(lat1, lon1, lat2_array, lon2_array):
    """
    Vectorized haversine distance (km) between a single point (lat1,lon1)
    and arrays lat2_array, lon2_array.
    """
    # convert to radians
    lat1r = radians(float(lat1))
    lon1r = radians(float(lon1))
    lat2r = np.radians(lat2_array.astype(float))
    lon2r = np.radians(lon2_array.astype(float))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_r = 6371.0
    return earth_r * c

def load_all_openaq(openaq_glob):
    files = sorted(glob.glob(openaq_glob))
    if not files:
        raise FileNotFoundError(f"No OpenAQ parquet files found in {openaq_glob}")
    dfs = []
    print(f"Found {len(files)} OpenAQ parquet files. Loading...")
    for f in files:
        print("  loading", os.path.basename(f))
        df = pd.read_parquet(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print("Combined OpenAQ rows:", len(combined))
    return combined

def standardize_openaq(df):
    # Ensure timestamp column named "ts" in UTC
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        # fallback to common names
        cand = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if cand:
            df["ts"] = pd.to_datetime(df[cand[0]], utc=True, errors="coerce")
        else:
            raise RuntimeError("No timestamp column found in OpenAQ data.")
    # Standardize station id / coords / parameter / value
    # station id fallbacks
    for col in ("station_id", "location", "station", "locationId", "city"):
        if col in df.columns:
            df["station_id"] = df[col]
            break
    # lat/lon fallbacks
    if "lat" not in df.columns and "latitude" in df.columns:
        df["lat"] = df["latitude"]
    if "lon" not in df.columns and "longitude" in df.columns:
        df["lon"] = df["longitude"]
    if "parameter" not in df.columns and "parameter" in df.columns:
        df["parameter"] = df["parameter"]  # noop
    # ensure value column
    if "value" not in df.columns:
        # try some common names
        for col in ("value", "measurement", "pm25", "pm2_5"):
            if col in df.columns:
                df["value"] = df[col]; break
    # drop rows without ts or value
    pre = len(df)
    df = df.dropna(subset=["ts","value"], how="any")
    print(f"Dropped {pre-len(df)} rows with missing ts/value from OpenAQ")
    # floor to hour for merging
    df["hour"] = df["ts"].dt.floor(TIME_FLOOR)
    return df

def load_firms(firms_glob):
    files = sorted(glob.glob(firms_glob))
    if not files:
        raise FileNotFoundError(f"No FIRMS parquet found in {firms_glob}")
    # if multiple FIRMS files, concat them
    dfs = []
    for f in files:
        print("Loading FIRMS:", os.path.basename(f))
        df = pd.read_parquet(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # standardize ts and lat/lon columns
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        # common acq_date + acq_time pattern
        if "acq_date" in df.columns and "acq_time" in df.columns:
            times = df["acq_time"].astype(str).str.zfill(4)
            df["ts"] = pd.to_datetime(df["acq_date"].astype(str) + " " + times, format="%Y-%m-%d %H%M", utc=True, errors="coerce")
        elif "acq_datetime" in df.columns:
            df["ts"] = pd.to_datetime(df["acq_datetime"], utc=True, errors="coerce")
        else:
            # try any date-like column
            cand = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if cand:
                df["ts"] = pd.to_datetime(df[cand[0]], utc=True, errors="coerce")
            else:
                raise RuntimeError("No timestamp column found in FIRMS data.")
    # normalize lat/lon
    if "latitude" in df.columns:
        df["lat"] = df["latitude"]
    if "longitude" in df.columns:
        df["lon"] = df["longitude"]
    # create hour floor
    df["hour"] = df["ts"].dt.floor(TIME_FLOOR)
    # ensure confidence & frp exist (may be NaN)
    if "confidence" not in df.columns:
        df["confidence"] = None
    if "frp" not in df.columns:
        df["frp"] = None
    print("FIRMS rows loaded:", len(df))
    return df

def load_meteo(meteo_glob):
    files = sorted(glob.glob(meteo_glob))
    if not files:
        raise FileNotFoundError(f"No METEO parquet found in {meteo_glob}")
    # if multiple, pick the first (or concat if needed)
    print("Loading METEO:", files[0])
    df = pd.read_parquet(files[0])
    # common column name is 'time' or 'ts'
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.rename(columns={"time":"ts"})
    elif "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        cand = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if cand:
            df["ts"] = pd.to_datetime(df[cand[0]], utc=True, errors="coerce")
        else:
            raise RuntimeError("No timestamp column found in METEO data.")
    df["hour"] = df["ts"].dt.floor(TIME_FLOOR)
    print("METEO rows loaded:", len(df))
    return df

def compute_station_hotspot_counts(stations_df, hotspots_df, radii_km):
    """
    For each station (station_id, lat, lon), compute per-hour hotspot counts
    for each radius in radii_km. Returns a DataFrame with columns:
      station_id, hour, count_r{r}_km (for each radius), sum_confidence_r{r}
    """
    # prepare hotspots arrays
    hotspots_df = hotspots_df.dropna(subset=["lat","lon","hour"]).copy()
    hotspots_df = hotspots_df.reset_index(drop=True)
    print("Hotspots rows (clean):", len(hotspots_df))
    results = []
    # iterate over stations (expected small, e.g., 10-50)
    stations = stations_df.dropna(subset=["lat","lon"]).drop_duplicates(subset=["station_id"])
    stations = stations.reset_index(drop=True)
    print("Stations to process:", len(stations))
    for idx, s in stations.iterrows():
        sid = s["station_id"]
        slat = s["lat"]
        slon = s["lon"]
        if pd.isna(slat) or pd.isna(slon):
            continue
        # compute distances to all hotspots (vectorized)
        dists = haversine_km(slat, slon, hotspots_df["lat"].values, hotspots_df["lon"].values)
        # append as column
        hotspots_df["_dist_km"] = dists
        # for each radius, filter hotspots and group by hour
        grouped = None
        agg_frames = []
        for r in radii_km:
            mask = hotspots_df["_dist_km"] <= r
            sel = hotspots_df[mask].copy()
            if sel.empty:
                # create empty df
                dfc = pd.DataFrame(columns=["hour", f"count_{int(r)}km", f"sum_conf_{int(r)}km"])
            else:
                grp = sel.groupby("hour").agg(
                    **{f"count_{int(r)}km": ("id" if "id" in sel.columns else sel.index.name or sel.index, "count"),
                       f"sum_conf_{int(r)}km": ("confidence","sum")}
                ).reset_index()
                # ensure column names exist
                if "hour" not in grp.columns:
                    grp = grp.reset_index().rename(columns={grp.columns[0]:"hour"})
                dfc = grp
            agg_frames.append(dfc)
        # merge all radius frames on hour
        from functools import reduce
        if agg_frames:
            merged = reduce(lambda left,right: pd.merge(left,right,on="hour", how="outer"), agg_frames)
        else:
            merged = pd.DataFrame(columns=["hour"])
        merged["station_id"] = sid
        # fill zeros
        for r in radii_km:
            merged[f"count_{int(r)}km"] = merged.get(f"count_{int(r)}km", 0).fillna(0).astype(int)
            merged[f"sum_conf_{int(r)}km"] = merged.get(f"sum_conf_{int(r)}km", 0).fillna(0)
        results.append(merged)
    if results:
        out = pd.concat(results, ignore_index=True)
        # ensure hour dtype is datetime UTC
        out["hour"] = pd.to_datetime(out["hour"], utc=True, errors="coerce")
        print("Produced station-hotspot counts rows:", len(out))
    else:
        out = pd.DataFrame(columns=["station_id","hour"])
    return out

def main():
    print("=== Phase 2: merging datasets ===")
    # Load OpenAQ
    openaq_raw = load_all_openaq(OPENAQ_GLOB)
    openaq = standardize_openaq(openaq_raw)
    # Keep pm25 and other params - for prototype we may filter to pm25 if desired:
    # openaq = openaq[openaq["parameter"]=="pm25"]

    # Create station lookup (unique station_id with lat/lon)
    station_coords = openaq[["station_id","lat","lon"]].drop_duplicates(subset=["station_id"]).reset_index(drop=True)
    station_coords = station_coords.dropna(subset=["lat","lon"])
    print("Unique stations with coords:", len(station_coords))

    # Load FIRMS
    firms = load_firms(FIRMS_FILENAME_GLOB)
    # ensure hotspots have id (if not, create)
    if "id" not in firms.columns:
        firms["id"] = firms.index.astype(str)

    # Compute station-hotspot hourly counts
    station_hot_counts = compute_station_hotspot_counts(station_coords, firms, RADII_KM)
    # Save hotspot counts (for debugging/inspection)
    shc_path = os.path.join(OUT_DIR, "station_hotspot_counts.parquet")
    # Save merged dataset
    print("Saving merged station-hotspot dataset...")
    # --- Fix data types before saving ---
    for col in station_hot_counts.columns:
        if station_hot_counts[col].dtype == 'object':
            station_hot_counts[col] = pd.to_numeric(station_hot_counts[col], errors='ignore')
    if 'sum_conf_10km' in station_hot_counts.columns:
        station_hot_counts['sum_conf_10km'] = pd.to_numeric(station_hot_counts['sum_conf_10km'], errors='coerce')

    station_hot_counts.to_parquet(shc_path, index=False)
    print("Saved station-hotspot counts to", shc_path)

    # Load Meteo
    meteo = load_meteo(METEO_FILENAME_GLOB)
    # choose columns to keep from meteo (all except ts/h)
    meteo_cols = [c for c in meteo.columns if c not in ("ts","hour")]
    # Reduce meteo to per-hour (if multiple rows per hour, take mean)
    meteo_hourly = meteo.groupby("hour")[meteo_cols].mean().reset_index()
    print("METEO hourly rows:", len(meteo_hourly))

    # Merge hotspot counts into readings (left join on station_id + hour)
# Merge station-hotspot counts into OpenAQ
# Load all OpenAQ parquet files
openaq_files = glob.glob(os.path.join(OPENAQ_FILENAME_GLOB))
openaq_list = [pd.read_parquet(f) for f in openaq_files]
openaq = pd.concat(openaq_list, ignore_index=True)
print("Loaded OpenAQ records:", len(openaq))

print("Merging OpenAQ + hotspot counts...")
# --- Ensure station_id types match before merging ---
openaq["station_id"] = openaq["station_id"].astype(str)
station_hot_counts["station_id"] = station_hot_counts["station_id"].astype(str)

merged = pd.merge(openaq, station_hot_counts, how="left", on=["station_id","hour"])
    # fill missing counts with zeros
for r in RADII_KM:
    merged[f"count_{int(r)}km"] = merged.get(f"count_{int(r)}km", 0).fillna(0).astype(int)
    merged[f"sum_conf_{int(r)}km"] = merged.get(f"sum_conf_{int(r)}km", 0).fillna(0)

    # Merge meteorology (left join on hour)
    merged = pd.merge(merged, meteo_hourly, how="left", on="hour")
    print("Merged dataset rows:", len(merged))

    # Save merged dataset
    out_path = os.path.join(OUT_DIR, "merged_dataset.parquet")
    merged.to_parquet(out_path, index=False)
    print("Saved merged dataset to:", out_path)
    # Print quick summary
    print("Sample rows:")
    print(merged.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
