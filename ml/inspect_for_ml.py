# inspect_for_ml.py
import os
import pandas as pd

PARQUET = os.path.join("..", "processed", "merged", "merged_dataset.parquet")

def main():
    print("Loading:", PARQUET)
    df = pd.read_parquet(PARQUET)
    print("Rows:", len(df))
    print("\nColumns:", df.columns.tolist())
    print("\nSample rows:")
    print(df.head(8).to_string(index=False))
    # show dtype summary
    print("\nDtypes:")
    print(df.dtypes)
    # Timestamp candidates
    ts_candidates = [c for c in df.columns if c.lower() in ("ts","time","timestamp","time_hour")]
    print("\nTimestamp candidates:", ts_candidates)
    # show null counts
    print("\nNull count per column:")
    print(df.isnull().sum().sort_values())
    # if parameter/value structure present, show unique parameters
    if "parameter" in df.columns and "value" in df.columns:
        print("\nUnique parameters (sample):", df["parameter"].dropna().unique()[:20])
        # how many rows have parameter==pm25
        print("Rows with parameter=='pm25':", len(df[df["parameter"].str.lower()=="pm25"]))
    # station_id / name check
    print("\nstation_id present:", "station_id" in df.columns)
    print("name present:", "name" in df.columns)
    if "station_id" in df.columns:
        print("Unique station_ids:", df["station_id"].nunique())
        print("Sample station_ids:", df["station_id"].unique()[:10])
    # timestamp range if exists
    for col in ts_candidates:
        try:
            s = pd.to_datetime(df[col], utc=True)
            print(f"\nRange for {col}: {s.min()} -> {s.max()}, freq estimate (head diffs):")
            diffs = s.sort_values().diff().dropna().value_counts().head(8)
            print(diffs)
        except Exception as e:
            print("Could not parse", col, e)

if __name__ == "__main__":
    main()
