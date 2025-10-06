import pandas as pd

parquet_path = r"C:\Users\SACHIN KUMAR\Desktop\airproj\PROCESSED\FIRMS VIIRS\SUOMI_VIIRS_C2_South_Asia_7d.parquet"

df = pd.read_parquet(parquet_path)
print(df.head())  # show first few rows
print(df.columns) # see all column names
print(df.info())  # see data types and size
