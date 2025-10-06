# fetch_and_clean_openaq_v3.py
"""
Fetch OpenAQ v3 measurements using an API key and save cleaned Parquet.

Usage:
  1) Put your API key in the API_KEY variable below.
  2) Run: python fetch_and_clean_openaq_v3.py
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# ========== CONFIG ==========
API_KEY = "64dfd2fb0a935eeef8f9432d3639a172faf290240ef4598c4549941beafa411b"   # <--- paste your key here (keep secret)
RAW_OPENAQ_DIR = r"C:\Users\SACHIN KUMAR\Desktop\airproj\RAW\OPEN AQ"
PROCESSED_OPENAQ_DIR = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\OPEN AQ"
CITY = "New Delhi"        # v3 often uses more specific city names; try "New Delhi" if "Delhi" fails
PARAMETER = "pm25"
DAYS = 30                 # pull past N days
PAGE_LIMIT = 10000        # v3 supports large page sizes
# ============================

os.makedirs(RAW_OPENAQ_DIR, exist_ok=True)
os.makedirs(PROCESSED_OPENAQ_DIR, exist_ok=True)

def get_headers():
    if not API_KEY or API_KEY.startswith("REPLACE"):
        raise ValueError("You must set API_KEY in this script (get it at https://explore.openaq.org/account).")
    return {"X-API-Key": API_KEY}

def fetch_openaq_v3(city, parameter, days, out_csv_path):
    # Use timezone-aware UTC times
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = "https://api.openaq.org/v3/measurements"
    params = {
        "city": city,
        "parameter": parameter,
        "date_from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": PAGE_LIMIT,
        "page": 1
    }
    rows = []
    headers = get_headers()
    print("Requesting OpenAQ v3:", params["date_from"], "->", params["date_to"])
    while True:
        r = requests.get(url, params=params, headers=headers, timeout=60)
        # If the API returns an error, raise a helpful message
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print("OpenAQ API error:", r.status_code, r.text[:300])
            raise

        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        rows.extend(results)
        print(f"Fetched page {params['page']} (records so far: {len(rows)})")
        # if fewer results than limit -> last page
        if len(results) < PAGE_LIMIT:
            break
        params["page"] += 1

    if not rows:
        print("No results returned for this query.")
        return None

    df = pd.json_normalize(rows)
    df.to_csv(out_csv_path, index=False)
    return df

def clean_and_save(df, out_parquet_path):
    # Create timezone-aware datetime column
    if "date.utc" in df.columns:
        df["ts"] = pd.to_datetime(df["date.utc"], utc=True)
    elif "date.local" in df.columns:
        df["ts"] = pd.to_datetime(df["date.local"], utc=True)
    else:
        # try generic parse of nested date
        if "date" in df.columns and df["date"].apply(lambda x: isinstance(x, dict)).all():
            df["ts"] = pd.to_datetime(df["date"].apply(lambda d: d.get("utc")), utc=True)
        else:
            # fallback - try parsing first date-like column
            date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if date_cols:
                df["ts"] = pd.to_datetime(df[date_cols[0]], utc=True)
            else:
                raise RuntimeError("No recognizable date column in OpenAQ response.")

    def safe_get(col):
        return df[col] if col in df.columns else None

    # gather coordinates robustly
    lat = safe_get("coordinates.latitude")
    lon = safe_get("coordinates.longitude")
    if lat is None and "coordinates" in df.columns:
        lat = df["coordinates"].apply(lambda x: x.get("latitude") if isinstance(x, dict) else None)
        lon = df["coordinates"].apply(lambda x: x.get("longitude") if isinstance(x, dict) else None)

    out = pd.DataFrame({
        "station_id": safe_get("location") if "location" in df.columns else safe_get("city"),
        "ts": df["ts"],
        "lat": lat,
        "lon": lon,
        "parameter": safe_get("parameter"),
        "value": safe_get("value"),
        "unit": safe_get("unit"),
        "source": "openaq_v3"
    })

    out = out.dropna(subset=["ts"]).drop_duplicates(subset=["station_id","ts","parameter"])
    out.to_parquet(out_parquet_path, index=False)
    print("Saved cleaned parquet:", out_parquet_path)
    return out

if __name__ == "__main__":
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_csv = os.path.join(RAW_OPENAQ_DIR, f"openaq_v3_{CITY}_{DAYS}d_{ts}.csv")
    parquet_out = os.path.join(PROCESSED_OPENAQ_DIR, f"openaq_v3_clean_{CITY}_{DAYS}d.parquet")
    df = fetch_openaq_v3(CITY, PARAMETER, DAYS, raw_csv)
    if df is not None:
        clean_and_save(df, parquet_out)
    else:
        print("No data fetched; check city name, API key, or query parameters.")
