import pandas as pd
import glob, os

# === FILE PATHS ===
BASE_DIR = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed"
OPENAQ_PATH = os.path.join(BASE_DIR, "OPEN AQ", "*.parquet")
FIRMS_PATH = os.path.join(BASE_DIR, "FIRMS VIIRS", "*.parquet")
METEO_PATH = os.path.join(BASE_DIR, "METEO", "*.parquet")
OUT_DIR = os.path.join(BASE_DIR, "merged")
os.makedirs(OUT_DIR, exist_ok=True)

def load_parquet_files(path_glob):
    files = glob.glob(path_glob)
    if not files:
        print(f"⚠️ No files found in: {path_glob}")
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"✅ Loaded {len(df)} rows from {len(files)} files: {path_glob}")
    return df

def main():
    # === 1️⃣ Load all data ===
    print("\nLoading OpenAQ data...")
    openaq = load_parquet_files(OPENAQ_PATH)
    print("Loading VIIRS hotspot data...")
    hotspots = load_parquet_files(FIRMS_PATH)
    print("Loading meteorology data...")
    meteo = load_parquet_files(METEO_PATH)

    # === 2️⃣ Process meteorology ===
    meteo["hour"] = pd.to_datetime(meteo["hour"])
    meteo_hourly = meteo.groupby("hour").mean().reset_index()
    print(f"Meteo hourly rows: {len(meteo_hourly)}")

    # === 3️⃣ Process hotspot counts ===
    if "station_id" not in openaq.columns:
        raise ValueError("OpenAQ data must include 'station_id' column.")
    if "hour" not in openaq.columns:
        raise ValueError("OpenAQ data must include 'hour' column.")
    if "hour" not in hotspots.columns:
        raise ValueError("FIRMS hotspot data must include 'hour' column.")

    # Group hotspot data by station_id + hour
    hotspot_counts = hotspots.groupby(["station_id", "hour"]).agg(
        count_1km=("count_1km", "sum"),
        sum_conf_1km=("sum_conf_1km", "sum"),
    ).reset_index()

    hotspot_out = os.path.join(OUT_DIR, "station_hotspot_counts.parquet")
    hotspot_counts.to_parquet(hotspot_out, index=False)
    print(f"✅ Saved hotspot summary: {hotspot_out}")

    # === 4️⃣ Merge OpenAQ + Hotspot data ===
    print("Merging OpenAQ + hotspot data...")
    openaq["station_id"] = openaq["station_id"].astype(str)
    hotspot_counts["station_id"] = hotspot_counts["station_id"].astype(str)
    merged = pd.merge(openaq, hotspot_counts, how="left", on=["station_id", "hour"])
    merged[["count_1km", "sum_conf_1km"]] = merged[["count_1km", "sum_conf_1km"]].fillna(0)

    # === 5️⃣ Merge with Meteo (on hour) ===
    print("Merging with meteorology data...")
    merged = pd.merge(merged, meteo_hourly, how="left", on="hour")

    print(f"✅ Final merged rows: {len(merged)}")

    # === 6️⃣ Save final dataset ===
    out_path = os.path.join(OUT_DIR, "merged_dataset.parquet")
    merged.to_parquet(out_path, index=False)
    print(f"✅ Saved merged dataset: {out_path}")

    # === 7️⃣ Preview ===
    print("\nSample preview:")
    print(merged.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
