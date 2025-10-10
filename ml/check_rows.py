import pandas as pd

df = pd.read_parquet('../processed/merged/merged_dataset.parquet')
df = df.sort_values('ts')
for lag in [1, 3, 6, 12, 24]:
    df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
for roll in [3, 6, 12, 24]:
    df[f'pm25_roll_{roll}'] = df['pm25'].rolling(roll, min_periods=1).mean()
df = df.fillna(method='ffill').dropna(subset=['pm25', 'lat', 'lon'])
print(len(df))