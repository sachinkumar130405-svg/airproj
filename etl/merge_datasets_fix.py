import os
import glob
import pandas as pd

# === CONFIGURATION ===
BASE_DIR = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed"

OPENAQ_PATH = os.path.join(BASE_DIR, "OPEN AQ", "*.parquet")
VIIRS_PATH = os.path.join(BASE_DIR, "FIRMS VIIRS", "*.parquet")
METEO_PATH = os.path.join(BASE_DIR, "METEO", "*.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "merged", "station_hotspot_counts.parquet")


def safe_read_parquets(path_pattern: str, label: str):
    """Reads multiple parquet files safely, returns concatenated dataframe."""
    files = glob.glob(path_pattern)
    if not files:
        print(f"⚠️ No {label} files found at {path_pattern}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Could not read {label} file {f}: {e}")

    if not dfs:
        print(f"⚠️ No valid {label} dataframes loaded.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    print(f"✅ Loaded {len(df)} rows from {len(files)} {label} files: {path_pattern}")
    return df


def main():
    print("\n=== STARTING DATA MERGE PIPELINE ===")

    # --- Load all datasets ---
    print("\nLoading OpenAQ data...")
    openaq = safe_read_parquets(OPENAQ_PATH, "OpenAQ")

    print("\nLoading VIIRS hotspot data...")
    viirs = safe_read_parquets(VIIRS_PATH, "VIIRS")

    print("\nLoading meteorology data...")
    meteo = safe_read_parquets(METEO_PATH, "METEO")

    if meteo.empty or openaq.empty or viirs.empty:
        print("⚠️ One or more datasets are empty. Check input folders before merging.")
        return

    # --- Fix column names ---
    meteo.columns = [c.lower().strip() for c in meteo.columns]
    openaq.columns = [c.lower().strip() for c in openaq.columns]
    viirs.columns = [c.lower().strip() for c in viirs.columns]

    # --- Handle datetime column in meteo ---
    possible_time_cols = [c for c in meteo.columns if "hour" in c or "time" in c or "date" in c]
    if possible_time_cols:
        time_col = possible_time_cols[0]
        print(f"🕒 Using '{time_col}' as timestamp column in METEO.")
        meteo["timestamp"] = pd.to_datetime(meteo[time_col], errors="coerce")
    else:
        print("⚠️ No 'hour' or time-like column found in METEO, creating dummy timestamp.")
        meteo["timestamp"] = pd.NaT

    # --- Merge datasets (example: simple merge logic, can be replaced with spatial joins) ---
    # For now, just group and count by station ID if available
    if "station_id" in openaq.columns:
        merged = (
            openaq.groupby("station_id")
            .size()
            .reset_index(name="air_quality_records")
        )
    else:
        merged = pd.DataFrame({"note": ["No 'station_id' found in OpenAQ dataset."]})
        print("⚠️ OpenAQ missing 'station_id'; skipping grouping.")

    # --- Save final output ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n💾 Merged dataset saved to: {OUTPUT_PATH}")
    print("✅ Merge pipeline completed successfully.")


if __name__ == "__main__":
    main()
