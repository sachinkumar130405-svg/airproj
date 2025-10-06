# convert_firms_viirs_csvs_to_parquet.py
"""
Batch-convert all CSV files in RAW\FIRMS VIIRS into cleaned Parquet files
in processed\FIRMS VIIRS.

This script handles typical FIRMS columns (acq_date, acq_time, latitude, longitude, confidence, brightness, frp).
Creates one Parquet per CSV input.
"""

import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --------- CONFIG: update if needed ----------
INPUT_DIR = r"C:\Users\SACHIN KUMAR\Desktop\airproj\RAW\FIRMS VIIRS"
OUTPUT_DIR = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\FIRMS VIIRS"
CHUNKSIZE = 200000
# ----------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_firms_chunk(df):
    # Typical FIRMS: acq_date (YYYY-MM-DD), acq_time (HHMM), latitude, longitude, confidence, brightness, frp
    if "acq_date" in df.columns and "acq_time" in df.columns:
        # ensure acq_time as str with 4 digits
        times = df["acq_time"].astype(str).str.zfill(4)
        dt = df["acq_date"].astype(str) + " " + times
        df["ts"] = pd.to_datetime(dt, format="%Y-%m-%d %H%M", errors="coerce", utc=True)
    elif "acq_datetime" in df.columns:
        df["ts"] = pd.to_datetime(df["acq_datetime"], utc=True, errors="coerce")
    else:
        # fallback: try to parse any datetime-like column
        date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if date_cols:
            df["ts"] = pd.to_datetime(df[date_cols[0]], utc=True, errors="coerce")
        else:
            df["ts"] = pd.NaT

    lat = df["latitude"] if "latitude" in df.columns else (df["lat"] if "lat" in df.columns else None)
    lon = df["longitude"] if "longitude" in df.columns else (df["lon"] if "lon" in df.columns else None)
    out = pd.DataFrame({
        "id": df.index.astype(str),
        "ts": df["ts"],
        "lat": lat,
        "lon": lon,
        "confidence": df.get("confidence"),
        "brightness": df.get("brightness"),
        "frp": df.get("frp"),
        "source": "firms_viirs"
    })
    out = out.dropna(subset=["ts", "lat", "lon"], how="any")
    return out

def convert_file(csv_path, parquet_path):
    print("Processing:", csv_path)
    reader = pd.read_csv(csv_path, chunksize=CHUNKSIZE, low_memory=False)
    writer = None
    written = 0
    for i, chunk in enumerate(reader):
        out_df = process_firms_chunk(chunk)
        if out_df.empty:
            continue
        table = pa.Table.from_pandas(out_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema)
        writer.write_table(table)
        written += len(out_df)
        print(f"  chunk {i+1}: wrote {len(out_df)} rows (total {written})")
    if writer:
        writer.close()
        print("Saved parquet:", parquet_path, " (approx rows):", written)
    else:
        print("No valid rows found in", csv_path)

def main():
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        print("No CSV files found in", INPUT_DIR)
        return
    for csv_path in csv_files:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_parquet = os.path.join(OUTPUT_DIR, f"{base}.parquet")
        convert_file(csv_path, out_parquet)

if __name__ == "__main__":
    main()
