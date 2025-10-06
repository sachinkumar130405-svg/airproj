# convert_openaq_csvs_to_parquet.py
"""
Batch-convert all CSV files in RAW\OPEN AQ into cleaned Parquet files in processed\OPEN AQ.
One Parquet file will be created per CSV input (same basename).

Adjust INPUT_DIR and OUTPUT_DIR to your paths if needed.
"""

import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import timezone
import pytz
import ast

# --------- CONFIG: update paths if needed ----------
INPUT_DIR = r"C:\Users\SACHIN KUMAR\Desktop\airproj\RAW\OPEN AQ"
OUTPUT_DIR = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\OPEN AQ"
CHUNKSIZE = 100000
TIMEZONE_LOCAL = "Asia/Kolkata"  # assume local times are India time if only local available
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_parse_coords(row):
    # try different column name patterns for lat/lon
    lat = None; lon = None
    for k in ("coordinates.latitude", "latitude", "lat"):
        if k in row: lat = row[k]; break
    for k in ("coordinates.longitude", "longitude", "lon", "long"):
        if k in row: lon = row[k]; break
    # sometimes coordinates stored as stringified dict in 'coordinates'
    if (lat is None or lon is None) and "coordinates" in row:
        try:
            val = row["coordinates"]
            if isinstance(val, str):
                parsed = ast.literal_eval(val)
            elif isinstance(val, dict):
                parsed = val
            else:
                parsed = None
            if isinstance(parsed, dict):
                lat = lat or parsed.get("latitude")
                lon = lon or parsed.get("longitude")
        except Exception:
            pass
    return lat, lon

def process_chunk(df):
    # create ts (timezone-aware UTC) using best available fields
    ts = None
    if "date.utc" in df.columns:
        ts = pd.to_datetime(df["date.utc"], utc=True, errors="coerce")
    elif "date.local" in df.columns:
        # assume local is India time if no tz info
        try:
            ts_local = pd.to_datetime(df["date.local"], errors="coerce")
            ts = ts_local.dt.tz_localize(TIMEZONE_LOCAL, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        except Exception:
            ts = pd.to_datetime(df["date.local"], utc=True, errors="coerce")
    elif "date" in df.columns and df["date"].apply(lambda x: isinstance(x, dict)).any():
        # nested date column: try to extract 'utc'
        try:
            df_dates = df["date"].apply(lambda d: d.get("utc") if isinstance(d, dict) else None)
            ts = pd.to_datetime(df_dates, utc=True, errors="coerce")
        except Exception:
            pass
    else:
        # fallback: try any column containing 'date' or 'time'
        date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if date_cols:
            ts = pd.to_datetime(df[date_cols[0]], utc=True, errors="coerce")

    # coordinates
    lat = None; lon = None
    # try common flattened names
    if "coordinates.latitude" in df.columns and "coordinates.longitude" in df.columns:
        lat = df["coordinates.latitude"]
        lon = df["coordinates.longitude"]
    elif "latitude" in df.columns and "longitude" in df.columns:
        lat = df["latitude"]
        lon = df["longitude"]
    elif "lat" in df.columns and "lon" in df.columns:
        lat = df["lat"]
        lon = df["lon"]
    else:
        # try parsing 'coordinates' col row-wise (slow fallback)
        lat_list = []
        lon_list = []
        if "coordinates" in df.columns:
            for v in df["coordinates"]:
                try:
                    if isinstance(v, dict):
                        lat_list.append(v.get("latitude"))
                        lon_list.append(v.get("longitude"))
                    elif isinstance(v, str):
                        parsed = ast.literal_eval(v)
                        lat_list.append(parsed.get("latitude"))
                        lon_list.append(parsed.get("longitude"))
                    else:
                        lat_list.append(None); lon_list.append(None)
                except Exception:
                    lat_list.append(None); lon_list.append(None)
            lat = pd.Series(lat_list)
            lon = pd.Series(lon_list)

    # station id
    station = None
    for c in ("location", "locationId", "station", "city"):
        if c in df.columns:
            station = df[c]; break

    # parameter/value/unit
    parameter = df["parameter"] if "parameter" in df.columns else None
    value = df["value"] if "value" in df.columns else None
    unit = df["unit"] if "unit" in df.columns else None

    out = pd.DataFrame({
        "station_id": station,
        "ts": ts,
        "lat": lat,
        "lon": lon,
        "parameter": parameter,
        "value": value,
        "unit": unit,
        "source": "openaq"
    })
    # drop rows without timestamp and without value
    out = out.dropna(subset=["ts", "value"], how="any")
    return out

def convert_file(csv_path, parquet_path):
    print("Processing:", csv_path)
    reader = pd.read_csv(csv_path, chunksize=CHUNKSIZE, low_memory=False)
    writer = None
    written = 0
    for i, chunk in enumerate(reader):
        out_df = process_chunk(chunk)
        if out_df.empty:
            continue
        # convert pandas -> pyarrow table
        table = pa.Table.from_pandas(out_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema)
        writer.write_table(table)
        written += len(out_df)
        print(f"  chunk {i+1}: wrote {len(out_df)} rows (total {written})")
    if writer:
        writer.close()
        print("Saved parquet:", parquet_path, " (total rows written approx):", written)
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
