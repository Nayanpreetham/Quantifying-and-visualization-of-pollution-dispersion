import xarray as xr
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

print("Loading dataset...")

ds = xr.open_dataset("processed/merged_dataset.nc")

AQI_VAR = list(ds.data_vars)[0]

X_list = []
y_list = []

# =========================
# BUILD DATASET
# =========================

for t in range(len(ds.time)-1):

    aqi = ds[AQI_VAR].isel(time=t).values
    y   = ds[AQI_VAR].isel(time=t+1).values

    features = np.stack([
        aqi,
        ds.u10.isel(time=t).values,
        ds.v10.isel(time=t).values,
        ds.wind_speed.isel(time=t).values,
        ds.pblh.isel(time=t).values,
        ds.t2m.isel(time=t).values,
        ds.sp.isel(time=t).values,
        ds.tp.isel(time=t).values,
        ds.elevation.values,
        ds.slope.values
    ], axis=-1)

    X_list.append(features.reshape(-1,10))
    y_list.append(y.reshape(-1))

X = np.concatenate(X_list)
y = np.concatenate(y_list)

print("Training Random Forest...")

# =========================
# MODEL
# =========================

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)

rf.fit(X, y)

# =========================
# SAVE
# =========================

os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/rf.pkl")

print("✅ Random Forest model saved at models/rf.pkl")