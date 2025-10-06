import pandas as pd
import glob

# Helper to preview a folder
def preview_parquet(folder_path):
    files = glob.glob(folder_path + "/*.parquet")
    if not files:
        print(f"\nNo files in {folder_path}")
        return
    df = pd.read_parquet(files[0])
    print(f"\n=== Preview: {folder_path} ===")
    print(f"File: {files[0]}")
    print(df.head(5))
    print("\nColumns:", df.columns.tolist())

# Inspect each folder
preview_parquet(r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\OPEN AQ")
preview_parquet(r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\FIRMS VIIRS")
preview_parquet(r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\METEO")
