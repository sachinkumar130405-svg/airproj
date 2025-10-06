"""
retrain_model.py
----------------
Automatically retrains the AQI prediction model if enough new data
has accumulated since the last model update.

Author: Sachin Kumar
Project: Air Quality + Route Optimization
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
MERGED_PATH = r"C:\Users\SACHIN KUMAR\Desktop\airproj\processed\merged\merged_dataset.parquet"
MODEL_PATH = r"C:\Users\SACHIN KUMAR\Desktop\airproj\ml\aqi_model_final.pkl"
MIN_NEW_ROWS = 200      # how many new rows trigger retraining
MAE_THRESHOLD = 25      # only accept new model if it's better than previous
BACKUP_MODELS = True


# --------------------------------------------------------
# LOAD EXISTING DATA & MODEL
# --------------------------------------------------------
def load_existing_model():
    if not os.path.exists(MODEL_PATH):
        print("⚠️ No existing model found — training a fresh one.")
        return None, None
    model = joblib.load(MODEL_PATH)
    model_mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
    return model, model_mtime


# --------------------------------------------------------
# RETRAIN MODEL LOGIC
# --------------------------------------------------------
def retrain_if_needed():
    print("🚀 Checking if retraining is required...")

    if not os.path.exists(MERGED_PATH):
        print("❌ Merged dataset not found.")
        return

    df = pd.read_parquet(MERGED_PATH)
    df = df.dropna(subset=["PM2.5", "PM10"])

    model, model_time = load_existing_model()

    if model_time:
        new_data = df[df["datetime"] > model_time]
        print(f"🕓 New data since last model: {len(new_data)} rows.")
        if len(new_data) < MIN_NEW_ROWS:
            print("⏸️ Not enough new data yet. Skipping retrain.")
            return
    else:
        new_data = df

    # Prepare features
    print("🧮 Preparing features...")
    X = df[["PM2.5", "PM10", "temperature_2m", "relativehumidity_2m",
            "windspeed_10m", "winddirection_10m", "precipitation"]].fillna(0)
    y = (0.5 * X["PM2.5"]) + (0.5 * X["PM10"])  # Simplified AQI proxy if target missing

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_new = RandomForestRegressor(n_estimators=120, random_state=42)
    model_new.fit(X_train, y_train)

    # Evaluate
    y_pred = model_new.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Retrained model — MAE: {mae:.2f}, R²: {r2:.3f}")

    # Compare with old model (if available)
    if model:
        y_pred_old = model.predict(X_test)
        mae_old = mean_absolute_error(y_test, y_pred_old)
        print(f"📊 Old model MAE: {mae_old:.2f}")

        if mae >= mae_old - 1:
            print("⚖️ New model not significantly better — keeping old model.")
            return
        elif mae > MAE_THRESHOLD:
            print("⚠️ Model performance too poor — not updating.")
            return

    # Backup old model
    if BACKUP_MODELS and os.path.exists(MODEL_PATH):
        backup_name = MODEL_PATH.replace(".pkl", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")
        os.rename(MODEL_PATH, backup_name)
        print(f"💾 Old model backed up: {backup_name}")

    # Save new model
    joblib.dump(model_new, MODEL_PATH)
    print(f"🎯 New model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    retrain_if_needed()

# after joblib.dump(new_model, MODEL_PATH)
import requests, os
API_RELOAD_URL = os.environ.get("API_RELOAD_URL", "http://127.0.0.1:8000/reload-model")
API_RELOAD_TOKEN = os.environ.get("RELOAD_TOKEN", "dev-token-change-me")
try:
    requests.post(API_RELOAD_URL, headers={"token": API_RELOAD_TOKEN}, timeout=5)
    print("Notified API to reload model")
except Exception as e:
    print("Warning: failed to notify API:", e)
