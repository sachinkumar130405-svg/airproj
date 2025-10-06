import pandas as pd
from pathlib import Path

# === Paths ===
MERGED_PATH = Path(r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\merged\merged_dataset.parquet")

print("=== Phase 3: Data Validation ===")

# === Load dataset ===
if not MERGED_PATH.exists():
    raise FileNotFoundError(f"❌ File not found: {MERGED_PATH}")

df = pd.read_parquet(MERGED_PATH)
print(f"✅ Loaded merged dataset: {MERGED_PATH.name} (rows={len(df):,}, columns={len(df.columns)})\n")

# === Basic structure ===
print("=== Columns ===")
print(df.columns.tolist(), "\n")

print("=== Sample rows ===")
print(df.head(5), "\n")

# === Time range ===
if "ts" in df.columns:
    df["ts"] = pd.to_datetime(df["ts"])
    print(f"📅 Time range: {df['ts'].min()} → {df['ts'].max()}\n")

# === Missing values ===
missing = df.isna().sum()
print("=== Missing values per column ===")
print(missing[missing > 0].sort_values(ascending=False), "\n")

# === Duplicates ===
dupes = df.duplicated().sum()
print(f"🔁 Duplicate rows: {dupes}\n")

# === Numeric summary ===
print("=== Numeric summary ===")
print(df.describe().T, "\n")

print("✅ Validation complete.")
