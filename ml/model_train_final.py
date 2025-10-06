#!/usr/bin/env python3
"""
Final training script:
- Loads merged parquets (parameter/value or wide)
- Builds features (lags, rolling, seasonal, interactions)
- Time-based holdout evaluation
- Optional LightGBM tuning with TimeSeriesSplit
- Retrains best model on ALL data and saves final model + metadata
"""
import os
import argparse
import warnings
from datetime import datetime
import math
import json

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------
# Utilities
# -------------------------
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
        return np.nan
    for cl, ch, il, ih in PM25_BREAKPOINTS:
        if cl <= pm25 <= ch:
            return int(round(((ih - il) / (ch - cl)) * (pm25 - cl) + il))
    return 500


def safe_makedirs(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -------------------------
# Data loading & normalization
# -------------------------
def load_merged_parquet(path_parquet):
    if not os.path.exists(path_parquet):
        raise FileNotFoundError(path_parquet)
    print("Loading parquet:", path_parquet)
    df = pd.read_parquet(path_parquet)
    # normalize timestamp column name
    for cand in ("ts", "time", "timestamp", "time_hour"):
        if cand in df.columns:
            df = df.rename(columns={cand: "ts"})
            break
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def pivot_parameter_value_if_needed(df):
    if "parameter" in df.columns and "value" in df.columns:
        print("Pivoting parameter/value -> wide table")
        df_param = df[["lat", "lon", "ts", "parameter", "value"]].copy()
        df_param["parameter"] = df_param["parameter"].str.lower()
        wide = (
            df_param.pivot_table(index=["lat", "lon", "ts"], columns="parameter", values="value", aggfunc="last")
            .reset_index()
        )
        # merge back available meta columns (e.g., fire_count, mean_frp, meteorology) by lat/lon/ts
        extra_cols = [c for c in df.columns if c not in ("lat", "lon", "parameter", "value", "ts")]
        if extra_cols:
            extras = df[["lat", "lon", "ts"] + [c for c in extra_cols if c in df.columns]].drop_duplicates(subset=["lat", "lon", "ts"])
            wide = pd.merge(wide, extras, on=["lat", "lon", "ts"], how="left")
        return wide
    else:
        return df.copy()


def create_station_ids(df, round_digits=5):
    df["lat_r"] = df["lat"].round(round_digits)
    df["lon_r"] = df["lon"].round(round_digits)
    df["station_coord"] = df["lat_r"].astype(str) + "_" + df["lon_r"].astype(str)
    mapping = {s: i for i, s in enumerate(df["station_coord"].unique())}
    df["station_id"] = df["station_coord"].map(mapping).astype(int)
    return df


def resample_by_station(df, freq="1h", ffill_limit=3):
    print("Resampling per station to freq=", freq)
    pollutant_cols = [c for c in ("pm25", "pm10", "no2", "so2", "co", "o3") if c in df.columns]
    meta_cols = [c for c in ("fire_count", "mean_frp", "temperature_2m", "relativehumidity_2m", "windspeed_10m", "winddirection_10m", "precipitation") if c in df.columns]
    out = []
    for sid, g in df.groupby("station_id", sort=False):
        g = g.sort_values("ts").set_index("ts")
        # numeric subset for aggregation
        numeric = g.select_dtypes(include=[np.number])
        # attempt to coerce lat/lon if present but not numeric
        if "lat" in g.columns and not np.issubdtype(g["lat"].dtype, np.number):
            numeric["lat"] = pd.to_numeric(g["lat"], errors="coerce")
        if "lon" in g.columns and not np.issubdtype(g["lon"].dtype, np.number):
            numeric["lon"] = pd.to_numeric(g["lon"], errors="coerce")
        # resample numeric by mean
        if numeric.shape[1] > 0:
            numeric_r = numeric.resample(freq).mean()
        else:
            numeric_r = pd.DataFrame(index=g.resample(freq).asfreq().index)
        # handle non-numeric columns: reindex and ffill/bfill small gaps
        other = g.drop(columns=numeric.columns, errors="ignore")
        if not other.empty:
            other_r = other.reindex(numeric_r.index).ffill(limit=ffill_limit).bfill(limit=ffill_limit)
        else:
            other_r = pd.DataFrame(index=numeric_r.index)
        combined = pd.concat([numeric_r, other_r], axis=1)
        # ensure lat/lon exist
        if "lat" not in combined.columns and "lat" in g.columns:
            combined["lat"] = g["lat"].ffill().bfill().reindex(combined.index)
        if "lon" not in combined.columns and "lon" in g.columns:
            combined["lon"] = g["lon"].ffill().bfill().reindex(combined.index)
        # forward fill pollutants small gaps
        for p in pollutant_cols:
            if p in combined.columns:
                combined[p] = combined[p].ffill(limit=ffill_limit)
        combined["station_id"] = sid
        combined = combined.reset_index()
        out.append(combined)
    if not out:
        return pd.DataFrame()
    df_res = pd.concat(out, ignore_index=True)
    print("Resampled rows:", len(df_res))
    return df_res


def ensure_aqi_col(df):
    if "aqi" not in df.columns and "pm25" in df.columns:
        print("Computing approximate AQI from pm25")
        df["aqi"] = df["pm25"].apply(lambda x: pm25_to_aqi(x) if not pd.isna(x) else np.nan)
    return df


# -------------------------
# Feature engineering
# -------------------------
def make_features(df, lags=(1, 3, 6, 24), rolling_windows=(3, 6, 24)):
    df = df.sort_values(["station_id", "ts"]).reset_index(drop=True)
    # time features
    df["hour"] = df["ts"].dt.hour
    df["dayofweek"] = df["ts"].dt.dayofweek
    df["month"] = df["ts"].dt.month
    df["weekofyear"] = df["ts"].dt.isocalendar().week.astype(int)
    # interaction features (if meteorology present)
    if "temperature_2m" in df.columns and "windspeed_10m" in df.columns:
        df["temp_x_wind"] = df["temperature_2m"] * df["windspeed_10m"]
    if "relativehumidity_2m" in df.columns and "temperature_2m" in df.columns:
        df["humidity_x_temp"] = df["relativehumidity_2m"] * df["temperature_2m"]
    # lags & rolling (pm25, pm10)
    for pollutant in ("pm25", "pm10"):
        if pollutant in df.columns:
            for lag in lags:
                df[f"{pollutant}_lag_{lag}"] = df.groupby("station_id")[pollutant].shift(lag)
            for rw in rolling_windows:
                df[f"{pollutant}_roll_{rw}"] = (
                    df.groupby("station_id")[pollutant].rolling(window=rw, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
                )
        else:
            # create placeholder columns to keep feature set stable
            for lag in lags:
                df[f"{pollutant}_lag_{lag}"] = np.nan
            for rw in rolling_windows:
                df[f"{pollutant}_roll_{rw}"] = np.nan
    # target is next-hour aqi
    df["aqi_target_1h"] = df.groupby("station_id")["aqi"].shift(-1)
    # drop rows with missing target
    df = df.dropna(subset=["aqi_target_1h"])
    # fill remaining numeric feature NaNs with medians (lenient)
    core_feats = [c for c in df.columns if any(k in c for k in ("pm25_lag_", "pm25_roll_", "pm10_lag_", "pm10_roll_"))]
    for c in core_feats:
        if c in df.columns:
            med = df[c].median()
            df[c] = df[c].fillna(med)
    return df


# -------------------------
# Training / tuning / final retrain
# -------------------------
def prepare_X_y(df, extra_features=None):
    feature_cols = [c for c in df.columns if c.startswith(("pm25_lag_", "pm25_roll_", "pm10_lag_", "pm10_roll_"))]
    # add seasonal/time features
    time_feats = ["hour", "dayofweek", "month", "weekofyear"]
    for tf in time_feats:
        if tf in df.columns:
            feature_cols.append(tf)
    # add interaction / meteorology features if present
    add_feats = ["temp_x_wind", "humidity_x_temp", "fire_count", "mean_frp", "temperature_2m", "windspeed_10m", "relativehumidity_2m"]
    for af in add_feats:
        if af in df.columns:
            feature_cols.append(af)
    # station encoding
    if "station_id" in df.columns:
        df["station_code"] = df["station_id"].astype(int)
        feature_cols.append("station_code")
    # dedupe feature columns keeping order
    feature_cols = [f for i, f in enumerate(feature_cols) if f and f not in feature_cols[:i]]
    X = df[feature_cols].astype(float)
    y = df["aqi_target_1h"].astype(float)
    return X, y, feature_cols


def train_with_optional_tuning(X_train, y_train, tune=True):
    if not tune or len(X_train) < 500:
        print("Skipping tuning (small data or --tune disabled). Training default LightGBM.")
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        return model, {"tuned": False}
    # TimeSeries CV
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {
        "num_leaves": [31, 63],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 500],
        "max_depth": [-1, 10],
        "subsample": [0.8, 1.0],
    }
    print("Starting GridSearchCV with TimeSeriesSplit (this may take a bit)...")
    base = lgb.LGBMRegressor(random_state=42)
    gscv = GridSearchCV(base, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    gscv.fit(X_train, y_train)
    print("GridSearch best params:", gscv.best_params_)
    best = gscv.best_estimator_
    # fit best on X_train
    best.fit(X_train, y_train)
    return best, {"tuned": True, "best_params": gscv.best_params_}


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    # baseline: persistence using pm25_lag_1 -> convert to aqi
    baseline_mae = None
    if "pm25_lag_1" in X_test.columns:
        baseline_aqi = X_test["pm25_lag_1"].apply(lambda p: pm25_to_aqi(p) if not pd.isna(p) else np.nan)
        mask = ~baseline_aqi.isna()
        if mask.sum() > 0:
            baseline_mae = mean_absolute_error(y_test[mask], baseline_aqi[mask])
    return {"mae": float(mae), "rmse": float(rmse), "baseline_mae": float(baseline_mae) if baseline_mae is not None else None}


# -------------------------
# Main flow
# -------------------------
def main(args):
    df_raw = load_merged_parquet(args.merged_parquet)
    df_wide = pivot_parameter_value_if_needed(df_raw)
    df_wide = create_station_ids(df_wide, round_digits=5)
    df_res = resample_by_station(df_wide, freq=args.resample_freq, ffill_limit=args.ffill_limit)
    df_res = ensure_aqi_col(df_res)

    # drop rows without ts or station_id
    df_res = df_res.dropna(subset=["ts", "station_id"]).reset_index(drop=True)
    print("After preprocessing rows:", len(df_res))

    df_feat = make_features(df_res)
    print("Feature rows available:", len(df_feat))
    if len(df_feat) == 0:
        raise RuntimeError("No feature rows available after feature engineering.")

    # Prepare X/y
    X_all, y_all, feature_list = prepare_X_y(df_feat)
    print("Using features:", feature_list)
    # time-based holdout: use last 20% as test
    df_feat_sorted = df_feat.sort_values("ts").reset_index(drop=True)
    split_i = int(len(df_feat_sorted) * (1 - args.test_frac))
    train_df = df_feat_sorted.iloc[:split_i]
    test_df = df_feat_sorted.iloc[split_i:]
    X_train, y_train, _ = prepare_X_y(train_df)
    X_test, y_test, _ = prepare_X_y(test_df)

    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)} (time-based holdout)")

    # tune/train on train set
    model, tune_info = train_with_optional_tuning(X_train, y_train, tune=args.tune)

    # evaluate on holdout
    eval_metrics = evaluate_model(model, X_test, y_test)
    print("Evaluation on holdout:", eval_metrics)

    # Now retrain on ALL data (train+test) to produce final artifact if requested
    print("Retraining best model on ALL data (train+test) to produce final deployed model...")
    model_full = model.__class__(**{k: v for k, v in getattr(model, "get_params", lambda: {})().items() if k in model.get_params().keys()})  # instantiate same class
    # fit on all
    model_full.fit(X_all, y_all)

    # save model + meta
    safe_makedirs(args.out_model)
    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "train_rows": int(len(X_all)),
        "feature_list": feature_list,
        "holdout_eval": eval_metrics,
        "tuning": tune_info,
        "script": os.path.basename(__file__),
    }
    joblib.dump({"model": model_full, "meta": meta}, args.out_model)
    print("Saved final model to:", args.out_model)
    print("Meta:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--merged-parquet", default=os.path.join("..", "processed", "merged", "merged_dataset.parquet"))
    p.add_argument("--out-model", default=os.path.join(os.path.dirname(__file__), "aqi_model_final.pkl"))
    p.add_argument("--tune", action="store_true", help="Run grid search tuning (time-series CV).")
    p.add_argument("--resample-freq", default="1h", help="Resample frequency (e.g. '1h').")
    p.add_argument("--ffill-limit", type=int, default=3, help="Forward-fill limit when resampling.")
    p.add_argument("--test-frac", type=float, default=0.2, help="Fraction of data kept as time-ordered holdout for evaluation.")
    args = p.parse_args()
    main(args)
