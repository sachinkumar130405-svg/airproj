import pandas as pd
import os
from pathlib import Path

# Path to your RAW folder
RAW_DIR = Path(r'C:\Users\SACHIN KUMAR\Desktop\airproj\RAW')

# Function to get column names from a CSV (reads first row only)
def get_columns(file_path):
    try:
        df = pd.read_csv(file_path, nrows=1)
        return list(df.columns)
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"

# List and print columns for CPCB files
cpcb_files = [f for f in os.listdir(RAW_DIR) if f.startswith('cpcb_dly_aq_delhi-') and f.endswith('.csv')]
print("CPCB Files Columns:")
for f in cpcb_files:
    columns = get_columns(RAW_DIR / f)
    print(f"{f}: {columns}")

# List and print columns for OpenAQ files
openaq_dir = RAW_DIR / 'OPEN AQ'
if os.path.exists(openaq_dir):
    openaq_files = [f for f in os.listdir(openaq_dir) if f.endswith('.csv')]
    print("\nOpenAQ Files Columns:")
    for f in openaq_files:
        columns = get_columns(openaq_dir / f)
        print(f"{f}: {columns}")

# List and print columns for other CSVs in RAW
other_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv') and not f.startswith('cpcb_dly_aq_delhi-') and 'openaq' not in f.lower()]
print("\nOther CSV Files Columns:")
for f in other_files:
    columns = get_columns(RAW_DIR / f)
    print(f"{f}: {columns}")