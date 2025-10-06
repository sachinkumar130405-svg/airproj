"""
Robust merge: OpenAQ (point readings) + FIRMS (hotspots) + METEO (hourly)
Matches by time (hour, +/- window) and by spatial proximity (km).

Save as: merge_datasets_geo_time_v2.py
Run:
& "C:/Users/SACHIN KUMAR/AppData/Local/Programs/Python/Python313/python.exe" "C:/Users/SACHIN KUMAR/Desktop/airproj/etl/merge_datasets_geo_time_v2.py"
"""

import glob, os
import pandas as pd
import numpy as np
from datetime import timedelta

# CONFIG - adjust if needed
OPENAQ_PATH = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\OPEN AQ\*.parquet"
FIRMS_PATH = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\FIRMS VIIRS\*.parquet"
METEO_PATH = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\METEO\*.parquet"
OUT_PATH = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\merged\merged_dataset.parquet"

# Parameters for matching
TIME_WINDOW_HOURS = 1        # look +/- this many hours around each reading
DISTANCE_KM = 50.0           # spatial radius to consider nearby hotspots (50 km)
PARAM_FILTER = None          # e.g. "pm25" to filter OpenAQ to pm2.5 only, or None

# ---------------- utility: haversine ----------------
def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    # returns numpy array of distances (km) between (lat1,lon1) and each point in lat2_arr/lon2_arr
    lat1r = np.radians(float(lat1))
    lon1r = np.radians(float(lon1))
    lat2r = np.radians(lat2_arr.astype(float))
    lon2r = np.radians(lon2_arr.astype(float))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c

# ---------------- load helper ----------------
def load_parquets_glob(path_glob, label):
    files = glob.glob(path_glob)
    if not files:
        print(f"[WARN] no {label} files found at {path_glob}")
        return pd.DataFrame()
    dfs = []
    for p in files:
        try:
            df = pd.read_parquet(p)
            dfs.append(df)
            print(f"[LOAD] {label}: {os.path.basename(p)} rows={len(df)}")
        except Exception as e:
            print(f"[ERROR] reading {p}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# ---------------- main merging logic ----------------
def main():
    print("=== merge_datasets_geo_time_v2 starting ===")
    openaq = load_parquets_glob(OPENAQ_PATH, "OpenAQ")
    firms = load_parquets_glob(FIRMS_PATH, "FIRMS")
    meteo = load_parquets_glob(METEO_PATH, "METEO")

    if openaq.empty:
        print("[ERROR] No OpenAQ data found. Aborting.")
        return
    if meteo.empty:
        print("[WARN] No METEO data found; proceeding without meteorology.")
    if firms.empty:
        print("[WARN] No FIRMS data found; merged dataset will not include hotspot counts.")

    # optional: filter to pm25 if desired
    if PARAM_FILTER:
        if "parameter" in openaq.columns:
            openaq = openaq[openaq["parameter"].astype(str).str.lower() == PARAM_FILTER.lower()].copy()
            print(f"[INFO] filtered OpenAQ to parameter={PARAM_FILTER}, rows now {len(openaq)}")
        else:
            print("[WARN] OpenAQ has no 'parameter' column; skipping parameter filter.")

    # ensure lat/lon exist
    for df, name in [(openaq, "OpenAQ"), (firms, "FIRMS")]:
        if df.empty:
            continue
        if "lat" not in df.columns and "latitude" in df.columns:
            df["lat"] = df["latitude"]
        if "lon" not in df.columns and "longitude" in df.columns:
            df["lon"] = df["longitude"]

    # floor times to hourly using lowercase 'h'
    openaq["ts_hour"] = pd.to_datetime(openaq["ts"], utc=True, errors="coerce").dt.floor("h")
    if not firms.empty:
        firms["ts_hour"] = pd.to_datetime(firms["ts"], utc=True, errors="coerce").dt.floor("h")
    if not meteo.empty:
        # meteo may have column named 'time' — keep it as 'meteo_time'
        time_col = "time" if "time" in meteo.columns else ("ts" if "ts" in meteo.columns else None)
        if time_col:
            meteo["time_hour"] = pd.to_datetime(meteo[time_col], utc=True, errors="coerce").dt.floor("h")
        else:
            # try to detect any datetime-like column
            parsed = None
            for c in meteo.columns:
                if "date" in c.lower() or "time" in c.lower() or "ts" in c.lower():
                    parsed = c; break
            if parsed:
                meteo["time_hour"] = pd.to_datetime(meteo[parsed], utc=True, errors="coerce").dt.floor("h")
            else:
                meteo["time_hour"] = pd.NaT

    # Precompute unique hours in OpenAQ to loop over (fewer iterations)
    unique_hours = sorted(openaq["ts_hour"].dropna().unique())
    print(f"[INFO] OpenAQ unique hourly bins: {len(unique_hours)}")

    merged_rows = []
    # for performance: group firms by hour into a dict of hour -> df
    firms_by_hour = {}
    if not firms.empty:
        # Create set of hours present in firms
        firms_hours = sorted(firms["ts_hour"].dropna().unique())
        for h in firms_hours:
            firms_by_hour[pd.Timestamp(h)] = firms[firms["ts_hour"] == h].reset_index(drop=True)

    # loop hours, match each OpenAQ reading in that hour to FIRMS in time window +/- TIME_WINDOW_HOURS
    print(f"[INFO] Matching hotspots within ±{TIME_WINDOW_HOURS} hour and {DISTANCE_KM} km")

    for hour in unique_hours:
        aq_group = openaq[openaq["ts_hour"] == hour]
        if aq_group.empty:
            continue
        # collect firms in time window
        firms_subset_frames = []
        if not firms.empty:
            for dt in range(-TIME_WINDOW_HOURS, TIME_WINDOW_HOURS + 1):
                hh = pd.Timestamp(hour) + pd.Timedelta(hours=dt)
                if hh in firms_by_hour:
                    firms_subset_frames.append(firms_by_hour[hh])
        if firms_subset_frames:
            firms_subset = pd.concat(firms_subset_frames, ignore_index=True)
        else:
            firms_subset = pd.DataFrame(columns=firms.columns) if not firms.empty else pd.DataFrame()

        # If no firms in window: set zero counts quickly
        if firms_subset.empty:
            for _, aq in aq_group.iterrows():
                merged_rows.append({
                    "ts": hour,
                    "lat": aq.get("lat"),
                    "lon": aq.get("lon"),
                    "parameter": aq.get("parameter"),
                    "value": aq.get("value"),
                    "fire_count": 0,
                    "mean_frp": 0.0
                })
            continue

        # for each aq point in this hour, compute distance to all firms in subset (vectorized per point)
        firms_lats = firms_subset["lat"].values
        firms_lons = firms_subset["lon"].values
        frp_vals = firms_subset["frp"].values if "frp" in firms_subset.columns else np.zeros(len(firms_subset))
        for _, aq in aq_group.iterrows():
            al = aq.get("lat"); ao = aq.get("lon")
            if pd.isna(al) or pd.isna(ao):
                # missing coords -> skip or add with zeros
                merged_rows.append({
                    "ts": hour,
                    "lat": al,
                    "lon": ao,
                    "parameter": aq.get("parameter"),
                    "value": aq.get("value"),
                    "fire_count": 0,
                    "mean_frp": 0.0
                })
                continue
            dists = haversine_km(al, ao, firms_lats, firms_lons)
            mask = dists <= DISTANCE_KM
            cnt = int(np.count_nonzero(mask))
            mean_frp = float(np.nanmean(frp_vals[mask])) if cnt > 0 else 0.0
            merged_rows.append({
                "ts": hour,
                "lat": al,
                "lon": ao,
                "parameter": aq.get("parameter"),
                "value": aq.get("value"),
                "fire_count": cnt,
                "mean_frp": mean_frp
            })

    merged_aq_fire = pd.DataFrame(merged_rows)
    print(f"[RESULT] merged OpenAQ+FIRMS rows: {len(merged_aq_fire)}")

    # If no merged rows (no nearby hotspots ever), fall back to saving OpenAQ + meteo join (no hotspot cols)
    if merged_aq_fire.empty:
        print("[WARN] No OpenAQ readings had FIRMS hotspots within the given time-window & distance.")
        # Create basic merged dataset by joining openaq (hourly) with meteo hourly (nearest)
        # Prepare meteo for merge_asof
        if not meteo.empty:
            meteo_hourly = meteo[["time_hour"] + [c for c in meteo.columns if c not in ("time", "time_hour")]].drop_duplicates(subset=["time_hour"]).sort_values("time_hour")
            # prepare openaq hourly
            openaq_hourly = openaq.copy().sort_values("ts_hour")
            # use merge_asof on hour
            merged_final = pd.merge_asof(
                openaq_hourly.sort_values("ts_hour"),
                meteo_hourly.sort_values("time_hour"),
                left_on="ts_hour",
                right_on="time_hour",
                direction="nearest"
            )
            # write out
            merged_final.to_parquet(OUT_PATH, index=False)
            print(f"[SAVE] Saved fallback merged dataset (OpenAQ + METEO) to {OUT_PATH}. Rows: {len(merged_final)}")
            return
        else:
            # Nothing to merge with meteorology or hotspots; save OpenAQ alone
            openaq.to_parquet(OUT_PATH, index=False)
            print(f"[SAVE] No FIRMS/METEO available. Saved raw OpenAQ to {OUT_PATH} (rows: {len(openaq)})")
            return

    # otherwise we have merged_aq_fire -> merge with meteo (nearest hour)
    if not meteo.empty:
        # prepare meteo_hourly
        meteo_hourly = meteo[["time_hour"] + [c for c in meteo.columns if c not in ("time", "time_hour")]].drop_duplicates(subset=["time_hour"]).sort_values("time_hour")
        merged_final = pd.merge_asof(
            merged_aq_fire.sort_values("ts"),
            meteo_hourly.sort_values("time_hour"),
            left_on="ts",
            right_on="time_hour",
            direction="nearest"
        )
    else:
        merged_final = merged_aq_fire.copy()

    # final save
    merged_final.to_parquet(OUT_PATH, index=False)
    print(f"[SAVE] Saved merged dataset to {OUT_PATH} (rows: {len(merged_final)})")
    print("Done.")

if __name__ == "__main__":
    main()
# At top of merge script, where Python path already set
from retrain_model import retrain_if_needed  # <- path must be importable (same folder or package)

# ... after you save merged dataset:
retrain_if_needed()

