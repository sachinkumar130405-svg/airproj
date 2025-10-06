"""
Robust trainer for AQI forecasting using merged parquets in parameter/value format.

What it does:
- Loads merged_dataset.parquet (parameter/value rows)
- Pivots parameter/value -> wide table (pm25, pm10, no2, so2, co, o3, etc.)
- Creates station_id from rounded (lat, lon)
- Resamples each station to hourly frequency, forward-fills small gaps
- Computes approximate AQI from PM2.5 if missing
- Builds lag + rolling features and next-hour AQI target
- Trains LightGBM and saves model+meta (joblib)
"""
import os, math
import pandas as pd
import numpy as np
from datetime import timedelta
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

# ------------ config ------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARQUET = os.path.join(ROOT, "processed", "merged", "merged_dataset.parquet")
OUT_MODEL = os.path.join(os.path.dirname(__file__), "aqi_forecast_from_merged.pkl")
PFILL_LIMIT = 3   # forward-fill limit (hours)
RESAMPLE_FREQ = "1H"
# ------------

# Simple approximate PM2.5 -> AQI (US EPA breakpoints)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(p):
    if pd.isna(p):
        return np.nan
    for cl, ch, il, ih in PM25_BREAKPOINTS:
        if cl <= p <= ch:
            return int(round(((ih - il) / (ch - cl)) * (p - cl) + il))
    return 500


def load_and_pivot(parquet_path):
    print("Loading parquet:", parquet_path)
    df = pd.read_parquet(parquet_path)
    # normalize ts column
    for c in ("ts", "time", "timestamp", "time_hour"):
        if c in df.columns:
            df = df.rename(columns={c: "ts"})
            break
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # ensure lat/lon exist
    if not {"lat", "lon"}.issubset(set(df.columns)):
        raise RuntimeError("lat/lon not found in merged file")

    # pivot parameter/value -> wide
    if "parameter" in df.columns and "value" in df.columns:
        print("Pivoting parameter/value to wide table...")
        df_param = df[["lat", "lon", "ts", "parameter", "value"]].copy()
        # lowercase parameter names
        df_param["parameter"] = df_param["parameter"].str.lower()
        # pivot using last value if duplicates
        wide = df_param.pivot_table(index=["lat", "lon", "ts"], columns="parameter", values="value", aggfunc="last").reset_index()
        # bring other useful columns (fire_count, mean_frp, meteo) by merging on (lat,lon,ts) using original df
        extra_cols = [c for c in df.columns if c not in ("lat", "lon", "parameter", "value", "ts")]
        extras = df[["lat", "lon", "ts"] + [c for c in extra_cols if c in df.columns]].drop_duplicates(subset=["lat", "lon", "ts"])
        merged = pd.merge(wide, extras, on=["lat", "lon", "ts"], how="left")
        # rename ts -> ts (already)
        df_wide = merged
    else:
        # maybe already wide
        df_wide = df.copy()

    print("Wide df columns:", list(df_wide.columns)[:30])
    print("Rows:", len(df_wide))
    return df_wide


def create_station_ids(df, round_digits=5):
    # round coordinates to stable strings to avoid floating noise
    df["lat_r"] = df["lat"].round(round_digits)
    df["lon_r"] = df["lon"].round(round_digits)
    # create station id string
    df["station_coord"] = df["lat_r"].astype(str) + "_" + df["lon_r"].astype(str)
    # map to integer ids for compactness
    mapping = {s: i for i, s in enumerate(df["station_coord"].unique())}
    df["station_id"] = df["station_coord"].map(mapping)
    print("Unique stations:", len(mapping))
    return df


def resample_and_ffill(df, freq=RESAMPLE_FREQ, ffill_limit=PFILL_LIMIT):
    """
    Resample each station to hourly/time freq. Handle non-numeric columns by
    resampling numeric columns with mean and reindexing + forward/backfilling
    non-numeric columns (like station_coord) separately.
    """
    out_rows = []
    # Pollutant and meta columns we may want to keep (if present)
    pollutant_cols = [c for c in ("pm25", "pm10", "no2", "so2", "co", "o3") if c in df.columns]
    meta_cols = [c for c in ("fire_count", "mean_frp", "temperature_2m",
                              "relativehumidity_2m", "windspeed_10m",
                              "winddirection_10m", "precipitation") if c in df.columns]

    # iterate stations
    for sid, g in df.groupby("station_id", sort=False):
        g = g.sort_values("ts").set_index("ts")

        # Drop completely-empty columns to avoid issues
        g = g.dropna(axis=1, how="all")

        # Select numeric columns for aggregation
        numeric = g.select_dtypes(include=[np.number])
        # If lat/lon are present but not numeric (unlikely) try to coerce
        if "lat" in g.columns and not np.issubdtype(g["lat"].dtype, np.number):
            numeric["lat"] = pd.to_numeric(g["lat"], errors="coerce")
        if "lon" in g.columns and not np.issubdtype(g["lon"].dtype, np.number):
            numeric["lon"] = pd.to_numeric(g["lon"], errors="coerce")

        # Resample numeric columns with mean
        if numeric.shape[1] > 0:
            numeric_r = numeric.resample(freq).mean()
        else:
            # No numeric columns -> create empty DataFrame with resampled index
            numeric_r = pd.DataFrame(index=g.resample(freq).asfreq().index)

        # Handle other (non-numeric) columns: reindex and forward/backfill
        other_cols = g.drop(columns=numeric.columns, errors="ignore")
        if not other_cols.empty:
            # Reindex other_cols to the resampled index and fill small gaps
            other_r = other_cols.reindex(numeric_r.index).ffill(limit=ffill_limit).bfill(limit=ffill_limit)
        else:
            other_r = pd.DataFrame(index=numeric_r.index)

        # Combine numeric and other columns
        combined = pd.concat([numeric_r, other_r], axis=1)

        # If lat/lon are missing after all, try to fill from original group (first non-null)
        if "lat" not in combined.columns and "lat" in g.columns:
            combined["lat"] = g["lat"].ffill().bfill().reindex(combined.index)
        if "lon" not in combined.columns and "lon" in g.columns:
            combined["lon"] = g["lon"].ffill().bfill().reindex(combined.index)

        # Fill small pollutant gaps with forward fill (limit), after aggregation
        for p in pollutant_cols:
            if p in combined.columns:
                combined[p] = combined[p].ffill(limit=ffill_limit)

        # Ensure station id present
        combined["station_id"] = sid

        out_rows.append(combined.reset_index())

    # Concatenate all stations back together
    if len(out_rows) == 0:
        return pd.DataFrame()  # nothing
    df_resampled = pd.concat(out_rows, ignore_index=True)

    print("Resampled total rows:", len(df_resampled))
    return df_resampled


def ensure_aqi(df):
    # If aqi missing but pm25 present, compute approximate aqi
    if "aqi" not in df.columns and "pm25" in df.columns:
        print("Computing approximate AQI from pm25")
        df["aqi"] = df["pm25"].apply(lambda x: pm25_to_aqi(x) if not pd.isna(x) else np.nan)
    return df


def build_features(df, lags=(1, 3, 6, 24), rolling=(3, 6, 24)):
    df = df.sort_values(["station_id", "ts"]).reset_index(drop=True)
    # create time features
    df["hour"] = df["ts"].dt.hour
    df["dayofweek"] = df["ts"].dt.dayofweek
    # pollutant lags + rolling for pm25 and pm10 if present
    for pollutant in ("pm25", "pm10"):
        if pollutant in df.columns:
            for lag in lags:
                df[f"{pollutant}_lag_{lag}"] = df.groupby("station_id")[pollutant].shift(lag)
            for rw in rolling:
                df[f"{pollutant}_roll_{rw}"] = df.groupby("station_id")[pollutant].rolling(window=rw, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
        else:
            # create NaN columns to keep consistent shape
            for lag in lags:
                df[f"{pollutant}_lag_{lag}"] = np.nan
            for rw in rolling:
                df[f"{pollutant}_roll_{rw}"] = np.nan
    # target: next hour aqi
    df["aqi_target_1h"] = df.groupby("station_id")["aqi"].shift(-1)
    # drop rows missing target
    df = df.dropna(subset=["aqi_target_1h"])
    # drop rows with NaN in core features (allow small amount by filling)
    feat_cols = [c for c in df.columns if ("pm25_lag" in c or "pm25_roll" in c or "pm10_lag" in c or "pm10_roll" in c)]
    # fill remaining NaNs with median per column (lenient)
    for c in feat_cols:
        if c in df.columns:
            med = df[c].median()
            df[c] = df[c].fillna(med)
    return df


def prepare_X_y(df):
    # pick features in order
    features = [c for c in df.columns if c.startswith(("pm25_lag_", "pm25_roll_", "pm10_lag_", "pm10_roll_"))]
    features += ["hour", "dayofweek"]
    # optional: station id code
    df["station_code"] = df["station_id"].astype(int)
    features.append("station_code")
    X = df[features]
    y = df["aqi_target_1h"]
    return X, y, features


def train_and_save(X, y, features, out_model_path):
    # simple train/test split (time-ordered)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # compute MAE and RMSE (compatibly across sklearn versions)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)   # MSE
    rmse = float(np.sqrt(mse))               # RMSE as sqrt(MSE)

    # persistence baseline using pm25_lag_1 if available
    baseline_mae = None
    if "pm25_lag_1" in X_test.columns:
        baseline_aqi = X_test["pm25_lag_1"].apply(lambda p: pm25_to_aqi(p) if not pd.isna(p) else np.nan)
        mask = ~baseline_aqi.isna()
        if mask.sum() > 0:
            baseline_mae = mean_absolute_error(y_test[mask], baseline_aqi[mask])

    meta = {"mae": float(mae), "rmse": float(rmse), "baseline_mae": float(baseline_mae) if baseline_mae is not None else None, "features": features}
    joblib.dump({"model": model, "meta": meta}, out_model_path)
    print("Saved model + meta to", out_model_path)
    print("MAE:", mae, "RMSE:", rmse, "Baseline MAE:", baseline_mae)
    return meta



def main():
    df_wide = load_and_pivot(PARQUET)
    df_wide = create_station_ids(df_wide, round_digits=5)
    df_res = resample_and_ffill(df_wide)
    df_res = ensure_aqi(df_res)
    # ensure ts column again
    df_res["ts"] = pd.to_datetime(df_res["ts"], utc=True)
    df_feat = build_features(df_res)
    print("Feature rows:", len(df_feat))
    if len(df_feat) == 0:
        raise RuntimeError("No feature rows available after feature engineering. Check data density and timestamps.")
    X, y, features = prepare_X_y(df_feat)
    meta = train_and_save(X, y, features, OUT_MODEL)
    print("Training finished. Meta:", meta)


if __name__ == "__main__":
    main()
