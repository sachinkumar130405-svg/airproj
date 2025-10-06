import pandas as pd
import glob
import numpy as np
from datetime import timedelta

# === CONFIG ===
OPENAQ_PATH = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\OPEN AQ\*.parquet"
FIRMS_PATH = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\FIRMS VIIRS\*.parquet"
METEO_PATH = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\METEO\*.parquet"
OUTPUT_PATH = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\merged\merged_dataset.parquet"

# === LOAD DATA ===
def load_parquets(path):
    files = glob.glob(path)
    if not files:
        print(f"⚠️ No files found in {path}")
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"✅ Loaded {len(df):,} rows from {len(files)} files: {path}")
    return df

print("\nLoading OpenAQ data...")
openaq = load_parquets(OPENAQ_PATH)
print("\nLoading VIIRS hotspot data...")
firms = load_parquets(FIRMS_PATH)
print("\nLoading meteorology data...")
meteo = load_parquets(METEO_PATH)

# === CLEAN / PREP ===
if openaq.empty or firms.empty or meteo.empty:
    print("❌ One or more datasets are empty. Please check your input folders.")
    exit()

# Harmonize timestamps
openaq["ts"] = pd.to_datetime(openaq["ts"]).dt.floor("H")
firms["ts"] = pd.to_datetime(firms["ts"]).dt.floor("H")
meteo["time"] = pd.to_datetime(meteo["time"]).dt.floor("H")

# === MERGE OPENAQ + FIRMS by time and proximity ===
def find_nearby_hotspots(aq_df, fire_df, dist_deg=0.1):
    print("\n🔄 Merging OpenAQ with nearby FIRMS hotspots...")
    merged_rows = []
    for hour, group in aq_df.groupby("ts"):
        fires = fire_df[fire_df["ts"] == hour]
        if fires.empty:
            continue
        for _, aq_row in group.iterrows():
            lat, lon = aq_row["lat"], aq_row["lon"]
            nearby = fires[
                (fires["lat"].between(lat - dist_deg, lat + dist_deg)) &
                (fires["lon"].between(lon - dist_deg, lon + dist_deg))
            ]
            merged_rows.append({
                "ts": hour,
                "lat": lat,
                "lon": lon,
                "parameter": aq_row["parameter"],
                "value": aq_row["value"],
                "fire_count": len(nearby),
                "mean_frp": nearby["frp"].mean() if not nearby.empty else 0
            })
    return pd.DataFrame(merged_rows)

merged_aq_fire = find_nearby_hotspots(openaq, firms)
print(f"✅ Produced {len(merged_aq_fire):,} merged rows (OpenAQ + FIRMS)")

# === MERGE WITH METEO (by nearest hour) ===
print("\n🔄 Adding meteorological data...")
merged_final = pd.merge_asof(
    merged_aq_fire.sort_values("ts"),
    meteo.sort_values("time"),
    left_on="ts",
    right_on="time",
    direction="nearest"
)

print(f"✅ Final merged rows: {len(merged_final):,}")

# === SAVE OUTPUT ===
merged_final.to_parquet(OUTPUT_PATH, index=False)
print(f"\n💾 Saved merged dataset to: {OUTPUT_PATH}")
print(f"📦 Size: ~{round(len(merged_final) * len(merged_final.columns) * 8 / 1024 / 1024, 2)} MB (approx)")
