import pandas as pd

# Path to your merged parquet file
merged_path = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\merged\station_hotspot_counts.parquet"

# Load the parquet file
df = pd.read_parquet(merged_path)

# Print basic info
print("\n=== File Info ===")
print(df.info())

# Show first 10 rows
print("\n=== Sample Rows ===")
print(df.head(10))

# Optional: show column names
print("\n=== Columns ===")
print(df.columns.tolist())
