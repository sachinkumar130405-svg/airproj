import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("=== Phase 4: Cleaning + Feature Engineering ===")

    merged_path = Path(r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\merged\merged_dataset.parquet")
    out_path = Path(r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\cleaned\cleaned_dataset.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- Load ----------------
    print(f"📂 Loading merged dataset from: {merged_path}")
    df = pd.read_parquet(merged_path)
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ---------------- Deduplication ----------------
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"🧹 Removed {before - after:,} duplicate rows")

    # ---------------- Datetime handling ----------------
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts"])
        df = df.sort_values("ts")
        df["year"] = df["ts"].dt.year
        df["month"] = df["ts"].dt.month
        df["day"] = df["ts"].dt.day
        df["hour"] = df["ts"].dt.hour
        df["dayofweek"] = df["ts"].dt.dayofweek

    # ---------------- Outlier cleaning ----------------
    if "value" in df.columns:
        q_low, q_high = df["value"].quantile([0.01, 0.99])
        df["value_clean"] = df["value"].clip(lower=q_low, upper=q_high)
        print(f"📊 Clipped outliers outside [{q_low:.2f}, {q_high:.2f}]")

    # ---------------- AQI categorization ----------------
    def categorize_aqi(v):
        if pd.isna(v): return "Unknown"
        if v <= 12: return "Good"
        if v <= 35.4: return "Moderate"
        if v <= 55.4: return "Unhealthy (Sensitive)"
        if v <= 150.4: return "Unhealthy"
        if v <= 250.4: return "Very Unhealthy"
        return "Hazardous"

    if "value_clean" in df.columns:
        df["aq_category"] = df["value_clean"].apply(categorize_aqi)

    # ---------------- Lag features ----------------
    if "ts" in df.columns and "value_clean" in df.columns:
        df["value_lag_1h"] = df["value_clean"].shift(1)
        df["value_change_1h"] = df["value_clean"] - df["value_lag_1h"]

    # ---------------- Handle missing after shifts ----------------
    df = df.fillna({
        "value_lag_1h": df["value_clean"].median(),
        "value_change_1h": 0
    })

    # ---------------- Save cleaned data ----------------
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved cleaned dataset to: {out_path}")
    print(f"📁 Rows: {len(df):,}, Columns: {len(df.columns)}")

    print("\n🎯 Cleaning & Feature Engineering Complete!")

if __name__ == "__main__":
    main()
