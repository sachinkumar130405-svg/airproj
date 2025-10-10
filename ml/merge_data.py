import pandas as pd
import os
from pathlib import Path
import numpy as np

# Station coordinates for CPCB stations in Delhi NCR
station_coords = {
    'Narela': (28.823, 77.102),
    'DTU': (28.749, 77.120),
    'Jahangirpuri': (28.733, 77.171),
    'Rohini': (28.733, 77.120),
    'Burari Crossing': (28.726, 77.201),
    'Sonia Vihar': (28.711, 77.249),
    'Wazirpur': (28.700, 77.165),
    'Ashok Vihar': (28.695, 77.182),
    'IHBAS Dilshad Garden': (28.681, 77.303),
    'Punjabi Bagh': (28.674, 77.131),
    'Vivek Vihar': (28.672, 77.315),
    'North Campus DU': (28.657, 77.159),
    'Shadipur': (28.651, 77.147),
    'Anand Vihar': (28.647, 77.316),
    'Pusa': (28.638, 77.169),
    'Mandir Marg': (28.636, 77.201),
    'ITO': (28.629, 77.241),
    'Patparganj': (28.624, 77.287),
    'Major Dhyan Chand National Stadium': (28.611, 77.238),
    'NSIT Dwarka': (28.609, 77.033),
    'Lodhi Road': (28.592, 77.227),
    'Jawaharlal Nehru Stadium': (28.580, 77.234),
    'Dwarka-Sector 8': (28.571, 77.072),
    'Najafgarh': (28.570, 76.934),
    'Nehru Nagar': (28.568, 77.251),
    'R K Puram': (28.563, 77.187),
    'IGI Airport (T3)': (28.563, 77.118),
    'CRRI Mathura Road': (28.551, 77.274),
    'Sirifort': (28.550, 77.216),
    'Okhla Phase-2': (28.531, 77.271),
    'Aya Nagar': (28.471, 77.110),
}

# Default Delhi lat/lon for aggregated data without coordinates
DEFAULT_DELHI_LAT = 28.61
DEFAULT_DELHI_LON = 77.20

# Root path
RAW_DIR = Path(__file__).parents[1] / "RAW"

# Load CPCB CSVs
cpcb_files = [f for f in os.listdir(RAW_DIR) if f.startswith('cpcb_dly_aq_delhi-') and f.endswith('.csv')]
dfs_cpcb = []
for f in cpcb_files:
    df = pd.read_csv(RAW_DIR / f)
    # Flexible column detection
    station_col = next((col for col in ['Location of Monitoring Station', 'City/Town/Village/Area', 'Stn Code'] if col in df.columns), None)
    date_col = next((col for col in ['Sampling Date'] if col in df.columns), None)
    pm25_col = next((col for col in ['PM 2.5'] if col in df.columns), None)
    pm10_col = next((col for col in ['RSPM/PM10'] if col in df.columns), None)

    if date_col and (pm25_col or pm10_col):
        # Use 'Location of Monitoring Station' or fallback to 'City/Town/Village/Area'
        df['station_name'] = df[station_col].str.strip() if station_col else 'Unknown'
        df['lat'] = df['station_name'].map(lambda s: station_coords.get(s, np.nan))
        df['lon'] = df['station_name'].map(lambda s: station_coords.get(s, np.nan))
        df['ts'] = pd.to_datetime(df[date_col], errors='coerce')
        df['pm25'] = pd.to_numeric(df[pm25_col], errors='coerce') if pm25_col else np.nan
        df['pm10'] = pd.to_numeric(df[pm10_col], errors='coerce') if pm10_col else np.nan
        df = df[['ts', 'lat', 'lon', 'pm25', 'pm10']].dropna(subset=['ts', 'lat', 'lon'])
        if not df.empty:
            dfs_cpcb.append(df)
    else:
        print(f"Skipping {f} due to missing expected columns")

df_cpcb = pd.concat(dfs_cpcb, ignore_index=True) if dfs_cpcb else pd.DataFrame()

# Load OpenAQ CSVs
openaq_dir = RAW_DIR / 'OPEN AQ'
if os.path.exists(openaq_dir):
    openaq_files = [f for f in os.listdir(openaq_dir) if f.endswith('.csv')]
    dfs_openaq = []
    for f in openaq_files:
        df = pd.read_csv(openaq_dir / f)
        lat_col = next((col for col in ['latitude'] if col in df.columns), None)
        lon_col = next((col for col in ['longitude'] if col in df.columns), None)
        time_col = next((col for col in ['datetimeUtc'] if col in df.columns), None)
        param_col = 'parameter' if 'parameter' in df.columns else None
        value_col = 'value' if 'value' in df.columns else None

        if lat_col and lon_col and time_col and param_col and value_col:
            df['lat'] = pd.to_numeric(df[lat_col], errors='coerce')
            df['lon'] = pd.to_numeric(df[lon_col], errors='coerce')
            df['ts'] = pd.to_datetime(df[time_col], errors='coerce')
            # Pivot parameter/value
            df = df.pivot(index=['lat', 'lon', 'ts'], columns=param_col, values=value_col).reset_index()
            df['pm25'] = pd.to_numeric(df.get('pm25', np.nan), errors='coerce')
            df['pm10'] = pd.to_numeric(df.get('pm10', np.nan), errors='coerce')
            df = df[['ts', 'lat', 'lon', 'pm25', 'pm10']].dropna(subset=['ts', 'lat', 'lon'])
            if not df.empty:
                dfs_openaq.append(df)
        else:
            print(f"Skipping {f} due to missing expected columns")

df_openaq = pd.concat(dfs_openaq, ignore_index=True) if dfs_openaq else pd.DataFrame()

# Load Other CSVs
other_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv') and not f.startswith('cpcb_dly_aq_delhi-') and 'openaq' not in f.lower() and 'RS_Session' not in f]
dfs_other = []
for f in other_files:
    df = pd.read_csv(RAW_DIR / f)
    if f == '3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.csv':
        # Handle 3b01... file
        lat_col = 'latitude' if 'latitude' in df.columns else None
        lon_col = 'longitude' if 'longitude' in df.columns else None
        time_col = 'last_update' if 'last_update' in df.columns else None
        param_col = 'pollutant_id' if 'pollutant_id' in df.columns else None
        value_col = 'pollutant_avg' if 'pollutant_avg' in df.columns else None

        if lat_col and lon_col and time_col and param_col and value_col:
            df['lat'] = pd.to_numeric(df[lat_col], errors='coerce')
            df['lon'] = pd.to_numeric(df[lon_col], errors='coerce')
            df['ts'] = pd.to_datetime(df[time_col], errors='coerce')
            # Pivot pollutant_id/pollutant_avg
            df = df.pivot(index=['lat', 'lon', 'ts'], columns=param_col, values=value_col).reset_index()
            df['pm25'] = pd.to_numeric(df.get('PM25', np.nan), errors='coerce')
            df['pm10'] = pd.to_numeric(df.get('PM10', np.nan), errors='coerce')
            df = df[['ts', 'lat', 'lon', 'pm25', 'pm10']].dropna(subset=['ts', 'lat', 'lon'])
            if not df.empty:
                dfs_other.append(df)

    elif f == 'delhi_aqi.csv':
        # Handle delhi_aqi.csv
        time_col = 'date' if 'date' in df.columns else None
        if time_col:
            df['ts'] = pd.to_datetime(df[time_col], errors='coerce')
            df['pm25'] = pd.to_numeric(df.get('pm2_5', np.nan), errors='coerce')
            df['pm10'] = pd.to_numeric(df.get('pm10', np.nan), errors='coerce')
            df['lat'] = DEFAULT_DELHI_LAT
            df['lon'] = DEFAULT_DELHI_LON
            df = df[['ts', 'lat', 'lon', 'pm25', 'pm10']].dropna(subset=['ts'])
            if not df.empty:
                dfs_other.append(df)

    elif f == 'delhi_pm25_aqi.csv':
        # Handle delhi_pm25_aqi.csv
        time_col = 'period.datetimeFrom.utc' if 'period.datetimeFrom.utc' in df.columns else None
        value_col = 'value' if 'value' in df.columns else None
        if time_col and value_col:
            df['ts'] = pd.to_datetime(df[time_col], errors='coerce')
            df['pm25'] = pd.to_numeric(df[value_col], errors='coerce')
            df['pm10'] = np.nan  # No pm10 here
            df['lat'] = DEFAULT_DELHI_LAT
            df['lon'] = DEFAULT_DELHI_LON
            df = df[['ts', 'lat', 'lon', 'pm25', 'pm10']].dropna(subset=['ts'])
            if not df.empty:
                dfs_other.append(df)
    else:
        print(f"Skipping {f} as it's not handled")

df_other = pd.concat(dfs_other, ignore_index=True) if dfs_other else pd.DataFrame()

# Final merge
df_merged = pd.concat([df_cpcb, df_openaq, df_other], ignore_index=True)
df_merged = df_merged.dropna(subset=['lat', 'lon', 'pm25'])  # Require at least pm25
df_merged.to_parquet(RAW_DIR / '../processed/merged/merged_dataset.parquet', index=False)
print("Merged data saved with", len(df_merged), "rows")