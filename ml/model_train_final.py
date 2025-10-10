import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from pathlib import Path
import joblib
import argparse
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description="Train AQI prediction model")
parser.add_argument('--merged-parquet', type=str, required=True, help='Path to merged parquet file')
parser.add_argument('--out-model', type=str, default='aqi_model_final.pkl', help='Output model file')
parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
args = parser.parse_args()

# Load data
data_path = Path(args.merged_parquet)
if not data_path.exists():
    raise FileNotFoundError(f"Merged parquet file not found at {data_path}")
df = pd.read_parquet(data_path)

# Feature engineering with minimal data loss
def create_features(df):
    df = df.sort_values('ts')
    df['hour'] = df['ts'].dt.hour
    df['dayofweek'] = df['ts'].dt.dayofweek
    df['month'] = df['ts'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Lags and rolls for pm25, pm10, and other pollutants
    for lag in [1, 3, 6, 12, 24]:
        df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
        df[f'pm10_lag_{lag}'] = df['pm10'].shift(lag)
    for roll in [3, 6, 12, 24]:
        df[f'pm25_roll_{roll}'] = df['pm25'].rolling(roll, min_periods=1).mean()
        df[f'pm10_roll_{roll}'] = df['pm10'].rolling(roll, min_periods=1).mean()
    
    # Include additional pollutants if available
    for col in ['SO2', 'NO2', 'SPM']:
        if col in df.columns:
            df[f'{col}_lag_1'] = df[col].shift(1)
            df[f'{col}_roll_3'] = df[col].rolling(3, min_periods=1).mean()
    
    # Forward fill to preserve data, then drop only if critical
    df = df.fillna(method='ffill').dropna(subset=['pm25', 'lat', 'lon'])
    return df

df = create_features(df)

# Prepare features and target
features = [col for col in df.columns if col not in ['ts', 'pm25', 'pm10']]
X = df[features].values
y = df['pm25'].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Time-based split with cross-validation
tscv = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(tscv.split(X))[-1]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Baseline: Use previous hour's value
y_baseline = np.roll(y_test, 1)
y_baseline[0] = y_test[0]  # Avoid shifting issue

# Model training with tuning
base_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 10,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'verbose': -1
}

if args.tune:
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 70],
        'feature_fraction': [0.7, 0.9],
        'max_depth': [10, 15]
    }
    model = GridSearchCV(
        LGBMRegressor(**base_params),
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        verbose=1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"Best parameters: {model.best_params_}")
    model = model.best_estimator_
else:
    model = LGBMRegressor(**base_params)
    model.fit(X_train, y_train)

# Recursive multi-step prediction for 24h
def predict_recursive(model, X, steps=24):
    predictions = []
    current_X = X.copy()
    for _ in range(steps):
        pred = model.predict(current_X)
        predictions.append(pred)
        # Update lagged features
        for lag in [1, 3, 6, 12, 24]:
            if f'pm25_lag_{lag}' in df.columns:
                current_X[:, features.index(f'pm25_lag_{lag}')] = np.roll(current_X[:, features.index(f'pm25_lag_{lag}')], -1)
                current_X[-1, features.index(f'pm25_lag_{lag}')] = pred[-1]
        # Update roll features (approximate)
        for roll in [3, 6, 12, 24]:
            if f'pm25_roll_{roll}' in df.columns:
                roll_idx = features.index(f'pm25_roll_{roll}')
                lag_cols = [features.index(f'pm25_lag_{i}') for i in range(1, roll+1) if f'pm25_lag_{i}' in features]
                if lag_cols:
                    current_X[-1, roll_idx] = np.mean(current_X[-roll:, lag_cols].mean(axis=1))
    return np.array(predictions).T

y_pred_1h = model.predict(X_test)
y_pred_24h = predict_recursive(model, X_test)

# Evaluate
mae_1h = mean_absolute_error(y_test[:len(y_pred_1h)], y_pred_1h)
rmse_1h = np.sqrt(mean_squared_error(y_test[:len(y_pred_1h)], y_pred_1h))
mae_24h = mean_absolute_error(y_test[:len(y_pred_24h)], y_pred_24h.mean(axis=1))
rmse_24h = np.sqrt(mean_squared_error(y_test[:len(y_pred_24h)], y_pred_24h.mean(axis=1)))
mae_baseline = mean_absolute_error(y_test, y_baseline)

print(f"Evaluation on holdout: {{'mae_1h': {mae_1h:.1f}, 'rmse_1h': {rmse_1h:.1f}, 'mae_24h': {mae_24h:.1f}, 'rmse_24h': {rmse_24h:.1f}, 'baseline_mae': {mae_baseline:.1f}}}")

# Save model and metadata
meta = {'feature_list': features, 'mae_1h': mae_1h, 'rmse_1h': rmse_1h, 'mae_24h': mae_24h, 'rmse_24h': rmse_24h}
joblib.dump({'model': model, 'meta': meta, 'scaler': scaler}, args.out_model)
print(f"Model saved to {args.out_model}")