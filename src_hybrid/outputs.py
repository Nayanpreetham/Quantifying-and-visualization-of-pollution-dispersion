import numpy as np
import matplotlib.pyplot as plt
import torch

# Import the edge-weight function from your graph module
from graph import compute_edge_weights

# Constants (must match app_physics.py)
GRID_RES_KM = 25.0
TIME_STEP_SCALE = 0.8
BACKGROUND_AQI = 1.0

def compute_dispersion(target, AQI_prev, wind, coords, elev, edge_index):
    """
    Identifies top‑3 zones contributing AQI change at the target location.
    Uses the same flux logic as physics_transport for consistency.
    """
    AQI_t = torch.tensor(AQI_prev, dtype=torch.float32)
    
    # Compute edge weights (direction‑dependent)
    w = compute_edge_weights(coords, wind, elev, edge_index)
    src, dst = edge_index

    # Wind magnitude at each node
    wind_speed = torch.sqrt(wind[:, 0]**2 + wind[:, 1]**2)

    # Courant‑based flux limiter (identical to physics_transport)
    max_flux_fraction = (wind_speed[src] * TIME_STEP_SCALE * 3600.0) / (GRID_RES_KM * 1000.0)
    max_flux_fraction = torch.clamp(max_flux_fraction, max=0.5)

    flux_coeff = w * max_flux_fraction

    contributions = []
    for i in range(len(src)):
        if dst[i] == target:
            # Distance between source and target in km
            dist = np.linalg.norm(coords[src[i]] - coords[target]) * GRID_RES_KM
            # Flux arriving from source i (before distance decay, if any)
            effect = (AQI_t[src[i]] * flux_coeff[i]).item()
            # Optional exponential distance decay (tuneable)
            effect = effect * np.exp(-dist / 200.0)
            contributions.append({
                "zone": f"Zone-{int(src[i])}",
                "distance": dist,
                "effect": effect
            })

    # Sort by absolute effect magnitude
    contributions.sort(key=lambda x: -abs(x["effect"]))
    return contributions[:3]

def print_dispersion(effects):
    print("\n🌬️  NEARBY ZONE DISPERSION (top 3)")
    print("     Positive = adds AQI, Negative = removes")
    print(f"{'Zone':<12} {'Distance(km)':<15} {'Effect(AQI)'}")
    print("-" * 45)
    for e in effects:
        print(f"{e['zone']:<12} {e['distance']:<15.1f} {e['effect']:+.2f}")
    print("=" * 50)

def compute_influence(AQI_prev, AQI_new):
    delta = AQI_new - AQI_prev
    zones = 7
    zone_matrix = np.zeros((zones, zones))
    grid = delta.reshape(21, 21)
    for i in range(zones):
        for j in range(zones):
            block = grid[i*3:(i+1)*3, j*3:(j+1)*3]
            zone_matrix[i, j] = np.mean(block)
    return zone_matrix

def print_influence(matrix):
    print("\n📊 ZONE INFLUENCE MATRIX")
    for row in matrix:
        print(" ".join(f"{v:+6.1f}" for v in row))
    print("=" * 50)

def plot_arrows(wind, coords):
    x = coords[:, 1].numpy()
    y = coords[:, 0].numpy()
    # Scale wind components for visibility
    u = wind[:, 0].numpy() / 5.0
    v = wind[:, 1].numpy() / 5.0
    plt.figure(figsize=(8, 6))
    plt.quiver(x, y, u, v)
    plt.title("Wind‑Driven Pollution Dispersion")
    plt.xlabel("Longitude Index")
    plt.ylabel("Latitude Index")
    plt.show()