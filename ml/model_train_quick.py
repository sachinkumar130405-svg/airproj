# model_train_quick.py
import os, warnings
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

PARQUET = os.path.join("..","processed","merged","merged_dataset.parquet")
OUT_MODEL = "aqi_forecast_quick.pkl"

# Simple PM2.5->AQI approx
def pm25_to_aqi(pm25):
    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    if pd.isna(pm25):
        return None
    for cl, ch, il, ih in bps:
        if cl <= pm25 <= ch:
            aqi = ((ih - il) / (ch - cl)) * (pm25 - cl) + il
            return int(round(aqi))
    return 500

def load_and_prep(parquet_path):
    print("Loading", parquet_path)
    df = pd.read_parquet(parquet_path)
    # find ts
    for c in ["ts","time","timestamp","time_hour"]:
        if c in df.columns:
            df = df.rename(columns={c:"ts"})
            break
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # pivot parameter/value if needed
    if "parameter" in df.columns and "value" in df.columns and "pm25" not in df.columns:
        print("Pivoting parameter/value table to wide (may drop other params)...")
        # keep only typical pollutants rows and pivot (take last value per ts/station/param)
        df_p = df[["station_id","ts","parameter","value"]].dropna()
        df_p["parameter"] = df_p["parameter"].str.lower()
        pivot = df_p.pivot_table(index=["station_id","ts"], columns="parameter", values="value", aggfunc="last").reset_index()
        df = pivot  # continue with pivoted
    # ensure station_id
    if "station_id" not in df.columns:
        if "name" in df.columns:
            df["station_id"] = df["name"]
        else:
            df["station_id"] = df.index.astype(str)
    # ensure pm25 column
    if "pm25" not in df.columns and "value" in df.columns:
        df = df.rename(columns={"value":"pm25"})
    if "pm25" in df.columns:
        df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    # ensure aqi
    if "aqi" not in df.columns and "pm25" in df.columns:
        df["aqi"] = df["pm25"].apply(lambda x: pm25_to_aqi(x) if not pd.isna(x) else None)
    # keep only needed columns
    needed = ["station_id","ts","pm25","pm10","no2","so2","co","o3","aqi"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    return df[["station_id","ts","pm25","pm10","no2","so2","co","o3","aqi"]]

def resample_per_station(df, freq="H", ffill_limit=3):
    out = []
    for sid, g in df.groupby("station_id", sort=False):
        g = g.set_index("ts").sort_index()
        # resample hourly
        g = g.resample(freq).mean()
        # station id column
        g["station_id"] = sid
        # forward-fill small gaps for pollutants
        g[["pm25","pm10","no2","so2","co","o3"]] = g[["pm25","pm10","no2","so2","co","o3"]].ffill(limit=ffill_limit)
        # recompute aqi if missing and pm25 present
        if g["aqi"].isna().all() and "pm25" in g.columns:
            g["aqi"] = g["pm25"].apply(lambda x: pm25_to_aqi(x) if not pd.isna(x) else np.nan)
        out.append(g.reset_index())
    df2 = pd.concat(out, ignore_index=True)
    print("After resample rows:", len(df2))
    return df2

def build_features_lenient(df):
    # sort and create hour/day features
    df = df.sort_values(["station_id","ts"]).reset_index(drop=True)
    df["hour"] = df["ts"].dt.hour
    df["dayofweek"] = df["ts"].dt.dayofweek
    # lags 1,3,6
    for lag in (1,3,6):
        df[f"pm25_lag_{lag}"] = df.groupby("station_id")["pm25"].shift(lag)
    # rolling 3,6
    for wid in (3,6):
        df[f"pm25_roll_{wid}"] = df.groupby("station_id")["pm25"].rolling(window=wid, min_periods=1).mean().shift(1).reset_index(level=0,drop=True)
    # target: next hour aqi
    df["aqi_target_1h"] = df.groupby("station_id")["aqi"].shift(-1)
    # drop rows where target missing
    df = df.dropna(subset=["aqi_target_1h"])
    # fill small remaining NAs in features with median
    feat_cols = [c for c in df.columns if "pm25_lag" in c or "pm25_roll" in c] + ["hour","dayofweek"]
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    return df

def train_quick(df_feat):
    # features selection
    features = [c for c in df_feat.columns if c.startswith("pm25_lag_") or c.startswith("pm25_roll_")] + ["hour","dayofweek"]
    X = df_feat[features]
    y = df_feat["aqi_target_1h"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    baseline = X_test["pm25_lag_1"].apply(lambda p: pm25_to_aqi(p) if not pd.isna(p) else np.nan)
    mask = ~baseline.isna()
    baseline_mae = mean_absolute_error(y_test[mask], baseline[mask]) if mask.sum()>0 else None
    print("Model MAE:", round(mae,2), "Baseline MAE:", round(baseline_mae,2) if baseline_mae is not None else None)
    joblib.dump({"model":model,"features":features}, OUT_MODEL)
    print("Saved model to", OUT_MODEL)

def main():
    df = load_and_prep(PARQUET)
    print("Raw loaded rows:", len(df))
    df2 = resample_per_station(df, freq="H", ffill_limit=3)
    df_feat = build_features_lenient(df2)
    print("Feature rows after build:", len(df_feat))
    if len(df_feat)==0:
        print("Still no feature rows — data too sparse or timestamps wrong.")
        return
    train_quick(df_feat)

if __name__ == "__main__":
    main()
