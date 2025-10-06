"""
Robust training script for AQI forecasting (LightGBM).

- Attempts to read historical measurements from Postgres (measurements table).
- If not found, falls back to reading processed merged parquet (merged_dataset.parquet).
- Builds lag/rolling/time features and a target (next-hour AQI).
- Trains LightGBM regressor, compares vs persistence baseline, saves model + metadata.

Usage:
    python model_train.py --merged-parquet "../processed/merged/merged_dataset.parquet"
"""

import os
import argparse
import warnings
from datetime import timedelta

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

# Optional DB access
try:
    import psycopg2
except Exception:
    psycopg2 = None
    warnings.warn("psycopg2 not installed; DB fallback to parquet only.")

# ------------------------------
# Utility: PM2.5 -> approximate AQI (US EPA breakpoints)
# ------------------------------
# This is an approximate mapping; it is good for quick prototypes.
# For accurate AQI use a standard library or full formula.
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(pm25):
    if pd.isna(pm25):
        return None
    for (clow, chigh, ilow, ihigh) in PM25_BREAKPOINTS:
        if clow <= pm25 <= chigh:
            aqi = ((ihigh - ilow) / (chigh - clow)) * (pm25 - clow) + ilow
            return int(round(aqi))
    # above highest breakpoint:
    return 500


# ------------------------------
# Data loaders
# ------------------------------
def load_from_db(conn_params, max_rows=None):
    if psycopg2 is None:
        raise RuntimeError("psycopg2 not available")

    print("Attempting DB load...")

    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()

    # Try to read a measurements table if exists
    try:
        cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name='measurements';"
        )
        exists = cur.fetchone()
        if not exists:
            raise RuntimeError("measurements table not found")
    except Exception as e:
        cur.close()
        conn.close()
        raise RuntimeError("measurements table not found or error checking tables") from e

    # Prefer commonly named columns; adapt if necessary
    sql = """
    SELECT m.station_id, s.name as station_name,
           m.time AT TIME ZONE 'UTC' as ts,
           m.pm25, m.pm10, m.no2, m.so2, m.co, m.o3, m.aqi
    FROM measurements m
    LEFT JOIN stations s ON m.station_id = s.id
    ORDER BY m.time
    """
    df = pd.read_sql(sql, conn)
    conn.close()

    if len(df) == 0:
        raise RuntimeError("DB measurements table is empty")

    if max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from DB.")
    return df


def load_from_parquet(parquet_path, max_rows=None):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)
    print(f"Loading merged parquet fallback: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.reset_index(drop=True)
    # Expect merged has columns like: ts/time/time_hour, station_id, lat, lon, pm25/value, parameter, aqi, meteorology...
    # Normalize common names:
    # find timestamp column
    for ts_col in ["ts", "time", "timestamp", "time_hour"]:
        if ts_col in df.columns:
            df = df.rename(columns={ts_col: "ts"})
            break
    # find pm25 in various forms
    if "pm25" not in df.columns:
        if "parameter" in df.columns and "value" in df.columns:
            # filter parameter == 'pm25'
            df_pm25 = df[df["parameter"].str.lower() == "pm25"]
            # If parameter/value structure is repeated per row (one param per row) we need to pivot
            if len(df_pm25) > 0:
                df = df_pm25.copy()
                df = df.rename(columns={"value": "pm25"})
            else:
                # maybe merged already aggregated; try 'value' as pm25
                df = df.rename(columns={"value": "pm25"})
    # Ensure essential columns exist
    if "station_id" not in df.columns and "station" in df.columns:
        df = df.rename(columns={"station": "station_id"})
    if "station_id" not in df.columns and "name" in df.columns:
        df["station_id"] = df["name"]
    if "aqi" not in df.columns and "pm25" in df.columns:
        # compute approximate AQI from pm25
        df["aqi"] = df["pm25"].apply(lambda x: pm25_to_aqi(x) if not pd.isna(x) else None)
    # normalize ts to datetime
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from parquet.")
    return df


# ------------------------------
# Feature engineering
# ------------------------------
def build_features(df, lags=(1, 3, 6, 24), rolling_windows=(3, 6, 24)):
    """
    df must contain: ts (datetime), station_id, pm25, pm10, etc, aqi
    Builds lagged features and rolling means per station.
    """
    df = df.copy()
    if "ts" not in df.columns:
        raise RuntimeError("timestamp column 'ts' missing")

    df = df.sort_values(["station_id", "ts"]).reset_index(drop=True)
    # ensure numeric pollutant columns exist
    for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        if col not in df.columns:
            df[col] = np.nan

    # per-station group lagging
    feature_rows = []
    grouped = df.groupby("station_id", sort=False)
    for station, g in grouped:
        g = g.sort_values("ts").reset_index(drop=True)
        # basic time features
        g["hour"] = g["ts"].dt.hour
        g["dayofweek"] = g["ts"].dt.dayofweek

        # lags of pm25 (and other pollutants)
        for lag in lags:
            g[f"pm25_lag_{lag}"] = g["pm25"].shift(lag)
            g[f"pm10_lag_{lag}"] = g["pm10"].shift(lag)
        # rolling means
        for rw in rolling_windows:
            g[f"pm25_roll_{rw}"] = g["pm25"].rolling(window=rw, min_periods=1).mean().shift(1)
            g[f"pm10_roll_{rw}"] = g["pm10"].rolling(window=rw, min_periods=1).mean().shift(1)

        # target: next-hour aqi (shift -1)
        g["aqi_target_1h"] = g["aqi"].shift(-1)

        feature_rows.append(g)

    df_feat = pd.concat(feature_rows, ignore_index=True)
    # Drop rows where target is null
    df_feat = df_feat.dropna(subset=["aqi_target_1h"])
    # Optionally drop rows with too many NaNs in features
    feature_cols = [c for c in df_feat.columns if any(s in c for s in ["pm25_lag", "pm25_roll", "hour", "dayofweek"])]
    df_feat = df_feat.dropna(subset=feature_cols, how="any")
    return df_feat


# ------------------------------
# Training pipeline
# ------------------------------
def prepare_X_y(df_feat):
    # choose features
    features = []
    # pollutant lags and rolls
    for c in df_feat.columns:
        if c.startswith("pm25_lag_") or c.startswith("pm10_lag_") or c.startswith("pm25_roll_") or c.startswith("pm10_roll_"):
            features.append(c)
    # add time features
    features += ["hour", "dayofweek"]
    # station_id as categorical (optional)
    if "station_id" in df_feat.columns:
        df_feat["station_id_cat"] = df_feat["station_id"].astype("category").cat.codes
        features.append("station_id_cat")
    X = df_feat[features]
    y = df_feat["aqi_target_1h"]
    return X, y, features


def train_and_evaluate(X, y, features, out_model_path):
    # time-based split: sort by index (which preserves time order per station concatenation),
    # but better: split by timestamp percentile if available. Here we split by rows while preserving time ordering.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")

    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    # persistence baseline (next-hour = current hour's pm25->aqi or previous aqi if available)
    # Here baseline: predict current row's aqi (approx) as next
    baseline = X_test.get("pm25_lag_1") if "pm25_lag_1" in X_test.columns else None
    if baseline is not None:
        baseline_aqi = baseline.apply(lambda p: pm25_to_aqi(p) if not pd.isna(p) else np.nan)
        # drop nan alignment
        mask = ~baseline_aqi.isna()
        if mask.sum() > 0:
            baseline_mae = mean_absolute_error(y_test[mask], baseline_aqi[mask])
        else:
            baseline_mae = None
    else:
        baseline_mae = None

    print(f"Model MAE = {mae:.2f}, RMSE = {rmse:.2f}")
    if baseline_mae is not None:
        print(f"Baseline persistence MAE = {baseline_mae:.2f}")

    # Save model and metadata
    meta = {
        "features": features,
        "mae": float(mae),
        "rmse": float(rmse),
        "baseline_mae": float(baseline_mae) if baseline_mae is not None else None
    }
    joblib.dump({"model": model, "meta": meta}, out_model_path)
    print(f"Saved model+meta to {out_model_path}")
    return model, meta


# ------------------------------
# Main
# ------------------------------
def main(args):
    # Try DB then parquet fallback
    df = None
    if args.db_name and psycopg2 is not None:
        print("DB connection params provided, attempting DB load...")
        conn_params = {
            "host": args.db_host,
            "database": args.db_name,
            "user": args.db_user,
            "password": args.db_pass,
            "port": args.db_port,
        }
        try:
            df = load_from_db(conn_params, max_rows=args.max_rows)
        except Exception as e:
            print("DB load failed:", str(e))
            df = None

    if df is None:
        # fallback to parquet
        try:
            df = load_from_parquet(args.merged_parquet, max_rows=args.max_rows)
        except Exception as e:
            raise RuntimeError("Unable to load data from DB or parquet: " + str(e))

    # Normalize columns: ensure station_id & ts & pm25 & aqi exist
    if "station_id" not in df.columns:
        if "name" in df.columns:
            df["station_id"] = df["name"]
        else:
            df["station_id"] = df.index.astype(str)

    if "ts" not in df.columns:
        # attempt common fallbacks
        for alt in ["time", "time_hour", "timestamp"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "ts"})
                break
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # If 'aqi' missing and 'pm25' present, compute approximate
    if "aqi" not in df.columns and "pm25" in df.columns:
        print("aqi column missing: computing approximate AQI from pm25")
        df["aqi"] = df["pm25"].apply(lambda x: pm25_to_aqi(x) if not pd.isna(x) else None)

    # Many OpenAQ datasets store pollutant as parameter/value; handle that
    if "pm25" not in df.columns and ("parameter" in df.columns and "value" in df.columns):
        # pivot may be necessary; simple approach: keep rows where parameter==pm25
        pm25_rows = df[df["parameter"].str.lower() == "pm25"]
        if len(pm25_rows) > 0:
            df = pm25_rows.rename(columns={"value": "pm25"})
        else:
            df["pm25"] = np.nan

    # Build features
    df_feat = build_features(df)

    if len(df_feat) == 0:
        raise RuntimeError("No feature rows after build_features. Check data density and timestamps.")

    X, y, features = prepare_X_y(df_feat)
    out_model = args.out_model or os.path.join(os.path.dirname(__file__), "aqi_forecast_model.pkl")
    model, meta = train_and_evaluate(X, y, features, out_model)

    print("Training complete. Summary:")
    print(meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", default="5432")
    parser.add_argument("--db-name", default="airq")
    parser.add_argument("--db-user", default="airuser")
    parser.add_argument("--db-pass", default="airpass")
    parser.add_argument("--merged-parquet", default=os.path.join("..", "processed", "merged", "merged_dataset.parquet"))
    parser.add_argument("--out-model", default=os.path.join(os.path.dirname(__file__), "aqi_forecast_model.pkl"))
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows for quicker testing")
    args = parser.parse_args()
    main(args)
