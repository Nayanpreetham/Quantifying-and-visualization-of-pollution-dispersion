# src/graph.py

import torch
import numpy as np

def build_edges(n=21):
    edges = []
    for i in range(n):
        for j in range(n):
            src = i*n + j
            for di in [-1,0,1]:
                for dj in [-1,0,1]:
                    if di==0 and dj==0: continue
                    ni,nj = i+di, j+dj
                    if 0<=ni<n and 0<=nj<n:
                        dst = ni*n + nj
                        edges.append([src,dst])
    return torch.tensor(edges).T


def compute_edge_weights(coords, wind, elev, edge_index):

    src, dst = edge_index

    vec = coords[dst] - coords[src]

    # 🔥 convert degrees → km
    lat = coords[:,0]
    lat_rad = torch.deg2rad(lat)

    lat_scale = 111.0
    lon_scale = 111.0 * torch.cos(lat_rad)

    vec_km = vec.clone()
    vec_km[:,0] = vec[:,0] * lat_scale
    vec_km[:,1] = vec[:,1] * lon_scale[src]

    dist = torch.norm(vec_km, dim=1) + 1e-6

    d_unit = vec_km / dist.unsqueeze(1)

    wind_vec = wind[src]

    wind_dot = (wind_vec * d_unit).sum(dim=1).clamp(min=0)
    wind_norm = torch.norm(wind_vec, dim=1) + 1e-6

    W_wind = wind_dot / wind_norm

    dh = elev[dst] - elev[src]
    W_terrain = torch.exp(-torch.relu(dh)/1000)

    w = W_wind * W_terrain * torch.exp(-dist/800)

    # 🔥 NORMALIZE (CRITICAL)
    w_exp = torch.exp(w)
    sum_per_src = torch.zeros_like(w_exp)
    sum_per_src.index_add_(0, src, w_exp)

    w_norm = w_exp / (sum_per_src[src] + 1e-6)

    return w_norm