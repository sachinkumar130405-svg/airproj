"""
update_data.py
---------------
Fetches latest air quality + weather data for Delhi, merges it,
and appends to the existing merged parquet file used by the ML model.

Author: Sachin Kumar
Project: Air Quality + Route Optimization
"""

import pandas as pd
import requests
import os
from datetime import datetime, timedelta

# ------------------------------
# CONFIGURATION
# ------------------------------
CITY = "Delhi"
COUNTRY = "IN"
LATITUDE = 28.6139
LONGITUDE = 77.2090
LIMIT = 1000

MERGED_PATH = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\merged\merged_dataset.parquet"
BACKUP_PATH = MERGED_PATH.replace(".parquet", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet")

# ------------------------------
# FETCH AQ DATA (OpenAQ)
# ------------------------------
def fetch_aq_data():
    print("📡 Fetching air quality data from OpenAQ...")
    url = f"https://api.openaq.org/v2/measurements"
    params = {
        "country": COUNTRY,
        "city": CITY,
        "limit": LIMIT,
        "parameter": ["pm25", "pm10"],
        "sort": "desc",
        "order_by": "datetime"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json().get("results", [])

    if not data:
        print("⚠️ No air quality data returned.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df[["parameter", "value", "unit", "date", "coordinates"]]
    df["datetime"] = pd.to_datetime(df["date"].apply(lambda x: x["utc"]))
    df["latitude"] = df["coordinates"].apply(lambda x: x.get("latitude", None))
    df["longitude"] = df["coordinates"].apply(lambda x: x.get("longitude", None))
    df = df.pivot_table(index="datetime", columns="parameter", values="value").reset_index()
    df = df.rename(columns={"pm25": "PM2.5", "pm10": "PM10"})
    print(f"✅ Retrieved {len(df)} air quality records.")
    return df


# ------------------------------
# FETCH WEATHER DATA (Open-Meteo)
# ------------------------------
def fetch_weather_data():
    print("🌦️ Fetching weather data from Open-Meteo...")
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        "&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,precipitation"
        "&past_days=1"
        "&forecast_days=1"
        "&timezone=auto"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    print(f"✅ Retrieved {len(df)} weather records.")
    return df


# ------------------------------
# MERGE + SAVE
# ------------------------------
def merge_and_save(aq_df, weather_df):
    if aq_df.empty or weather_df.empty:
        print("❌ Skipping merge: missing data.")
        return

    print("🧩 Merging air and weather datasets...")
    df = pd.merge_asof(
        aq_df.sort_values("datetime"),
        weather_df.sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("1H")
    )

    if os.path.exists(MERGED_PATH):
        print("📂 Loading existing merged dataset...")
        existing = pd.read_parquet(MERGED_PATH)
        # Avoid duplicates by datetime
        df = pd.concat([existing, df]).drop_duplicates(subset=["datetime"]).sort_values("datetime")
        # Backup before overwriting
        existing.to_parquet(BACKUP_PATH)
        print(f"💾 Backup created at: {BACKUP_PATH}")

    df.to_parquet(MERGED_PATH, index=False)
    print(f"✅ Merged data saved successfully to: {MERGED_PATH}")
    print(f"Total records: {len(df)}")


# ------------------------------
# MAIN
# ------------------------------
def main():
    print("🚀 Running automated data update...")
    aq_df = fetch_aq_data()
    weather_df = fetch_weather_data()
    merge_and_save(aq_df, weather_df)
    print("🏁 Data update complete.")


if __name__ == "__main__":
    main()
