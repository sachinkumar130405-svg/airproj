# convert_firms_to_parquet.py
import os
import pandas as pd

RAW_FIRMS = r"C:\Users\SACHIN KUMAR\Desktop\airproj\RAW\FIRMS VIIRS\firms_viirs_7d.csv"
OUT_PARQUET = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\FIRMS VIIRS\firms_viirs_clean.parquet"

def clean_firms(raw_csv, out_parquet):
    df = pd.read_csv(raw_csv)
    # typical FIRMS CSV fields: acq_date, acq_time, latitude, longitude, confidence, version, brightness, frp, etc.
    # create timestamp (FIRMS often has acq_date and acq_time as HHMM integers)
    if 'acq_date' in df.columns and 'acq_time' in df.columns:
        timestr = df['acq_date'].astype(str) + " " + df['acq_time'].astype(str).str.zfill(4)
        df['ts'] = pd.to_datetime(timestr, format="%Y-%m-%d %H%M", utc=True)
    elif 'acq_datetime' in df.columns:
        df['ts'] = pd.to_datetime(df['acq_datetime'], utc=True)
    out = pd.DataFrame({
        'id': df.index.astype(str),
        'ts': df['ts'],
        'lat': df.get('latitude'),
        'lon': df.get('longitude'),
        'confidence': df.get('confidence'),
        'brightness': df.get('brightness'),
        'frp': df.get('frp'),
        'source': 'firms_viirs'
    })
    out = out.dropna(subset=['ts'])
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    print("Saved FIRMS parquet to", out_parquet)

if __name__ == "__main__":
    clean_firms(RAW_FIRMS, OUT_PARQUET)
