import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import xarray as xr
import joblib
import torch
import matplotlib.pyplot as plt

from graph import build_edges, compute_edge_weights
from outputs import (
    compute_dispersion,
    print_dispersion,
    plot_arrows,
    compute_influence,
    print_influence
)

# =========================
# CONFIGURATION
# =========================
LAT_MIN, LAT_MAX = 25, 30
LON_MIN, LON_MAX = 75, 80
BACKGROUND_AQI = 1.0           # Absolute minimum (clean air)
GRID_RES_KM = 25.0             # km per grid cell
TIME_STEP_SCALE = 0.8          # Fraction of Courant number (tuneable)

VALID_HORIZONS = {
    "15 days": 15,
    "1 month": 30,
    "3 months": 90,
    "6 months": 180,
    "1 year": 365
}

# =========================
# USER INPUT
# =========================
lat = float(input("Enter latitude: "))
lon = float(input("Enter longitude: "))
if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
    print("❌ Location out of bounds")
    exit()

h = input("Choose time horizon (15 days, 1 month, 3 months, 6 months, 1 year): ")
if h not in VALID_HORIZONS:
    print("❌ Invalid time horizon")
    exit()
steps = VALID_HORIZONS[h]

# =========================
# LOAD DATA & MODELS
# =========================
print("\nLoading data...")
ds = xr.open_dataset("processed/merged_dataset.nc")
xgb = joblib.load("models/xgb.pkl")
rf = joblib.load("models/rf.pkl")

AQI_VAR = list(ds.data_vars)[0]
AQI = ds[AQI_VAR].isel(time=-1).values.reshape(-1).astype(np.float32)

print(f"Initial AQI: min={AQI.min():.2f}, max={AQI.max():.2f}, mean={AQI.mean():.2f}")

# =========================
# GRAPH SETUP
# =========================
edge_index = build_edges()
coords = torch.tensor([[i // 21, i % 21] for i in range(441)], dtype=torch.float32)
elev = torch.tensor(ds.elevation.values.reshape(-1), dtype=torch.float32)

lat_arr = ds.lat.values
lon_arr = ds.lon.values
lat_idx = np.argmin(np.abs(lat_arr - lat))
lon_idx = np.argmin(np.abs(lon_arr - lon))
target = lat_idx * 21 + lon_idx

# =========================
# ROBUST PHYSICS TRANSPORT
# =========================
def physics_transport(AQI_in, wind, elev, debug=False):
    """
    Conservative upwind advection with diagnostics.
    Returns updated AQI (numpy array).
    """
    AQI_t = torch.tensor(AQI_in, dtype=torch.float32)
    
    # Compute edge weights (direction‑dependent)
    w = compute_edge_weights(coords, wind, elev, edge_index)
    src, dst = edge_index

    wind_speed = torch.sqrt(wind[:, 0]**2 + wind[:, 1]**2)
    
    # Courant‑based flux limiter
    # max_flux_fraction = (wind_speed * dt * scaling) / grid_length
    # Here dt is implicitly 1 hour (since we have hourly data)
    max_flux_fraction = (wind_speed[src] * TIME_STEP_SCALE * 3600.0) / (GRID_RES_KM * 1000.0)
    max_flux_fraction = torch.clamp(max_flux_fraction, max=0.5)   # ≤50% per step

    flux_coeff = w * max_flux_fraction
    available = AQI_t[src]
    flux = flux_coeff * available

    inflow = torch.zeros_like(AQI_t)
    outflow = torch.zeros_like(AQI_t)
    inflow.index_add_(0, dst, flux)
    outflow.index_add_(0, src, flux)

    AQI_new = AQI_t + inflow - outflow
    AQI_new = torch.clamp(AQI_new, min=BACKGROUND_AQI)

    if debug:
        print(f"   [Physics] w max={w.max():.4f}, flux_coeff max={flux_coeff.max():.4f}, "
              f"flux max={flux.max():.2f}, ΔAQI mean={(AQI_new - AQI_t).mean():.2f}")

    return AQI_new.numpy()

# =========================
# MAIN TIME LOOP WITH DIAGNOSTICS
# =========================
print("Running prediction...")
for t in range(min(steps, len(ds.time))):
    AQI_prev = AQI.copy()

    # Build features
    features = np.stack([
        AQI_prev.reshape(21, 21),
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
    X = features.reshape(-1, 10)

    # ML predictions
    aqi_xgb = xgb.predict(X)
    aqi_rf = rf.predict(X)
    AQI_ml = 0.7 * aqi_xgb + 0.3 * aqi_rf
    AQI_ml = np.maximum(AQI_ml, BACKGROUND_AQI)

    # Transport
    wind = torch.tensor(X[:, 1:3], dtype=torch.float32)
    AQI_trans = physics_transport(AQI_prev, wind, elev, debug=(t==0))

    # Blend
    AQI = AQI_ml + (AQI_trans - AQI_prev)
    AQI = np.maximum(AQI, BACKGROUND_AQI)

    if t == 0:
        print(f"Step {t:03d}: ML mean={AQI_ml.mean():.2f}, Transport mean={AQI_trans.mean():.2f}, "
              f"Δ mean={(AQI_trans - AQI_prev).mean():.2f}, Final mean={AQI.mean():.2f}")

# =========================
# OUTPUT
# =========================
final_AQI = AQI[target]
print("========================================")
print(f"📍 Location: ({lat:.2f}, {lon:.2f})")
print(f"🕒 Time horizon: {h}")
print("========================================")

# Use the last wind for visualisation (consistent with previous)
effects = compute_dispersion(target, AQI_prev, wind, coords, elev, edge_index)
print_dispersion(effects)

matrix = compute_influence(AQI_prev, AQI)
print_influence(matrix)

plot_arrows(wind, coords)
