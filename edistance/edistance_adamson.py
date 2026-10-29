import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

print("Start E-distance Analysis on Full Gene Expression (All Perturbations)")

DATASET = "adamson16"
CONTROL_LABEL = "control"

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
adata_full = ad.read_h5ad(DATA_PATH)
print(f"Full dataset shape: {adata_full.shape}")

# Extract full expression matrix (all genes, no HVG/PCA filtering)
print("Extracting full expression matrix...")
X_full = adata_full.X.toarray() if hasattr(adata_full.X, "toarray") else adata_full.X

# Get control cell indices and expression
ctrl_mask = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
ctrl_indices = np.where(ctrl_mask)[0]
Y_ctrl = X_full[ctrl_indices]
print(f"Number of control cells: {len(ctrl_indices)}")

# Function to compute Energy Distance (E-distance)
def compute_energy_distance(X, Y):
    n = X.shape[0]
    m = Y.shape[0]
    if n == 0 or m == 0:
        return 0.0
    
    d_xy = cdist(X, Y, metric='euclidean')
    d_xx = cdist(X, X, metric='euclidean')
    d_yy = cdist(Y, Y, metric='euclidean')
    
    term1 = 2.0 / (n * m) * np.sum(d_xy)
    term2 = 1.0 / (n**2) * np.sum(d_xx)
    term3 = 1.0 / (m**2) * np.sum(d_yy)
    
    d_sq = term1 - term2 - term3
    return np.sqrt(max(0.0, d_sq))

# Compute E-distance for all unique perturbations against control
all_perts = adata_full.obs["perturbation"].astype(str).unique()
perts = [p for p in all_perts if p != CONTROL_LABEL]
print(f"Computing E-distance for all {len(perts)} perturbations on full gene space...")

results = []
for pert in perts:
    pert_mask = (adata_full.obs["perturbation"] == pert).values
    n_pert = pert_mask.sum()
    
    if n_pert == 0:
        continue
        
    X_pert = X_full[pert_mask]
    edist = compute_energy_distance(X_pert, Y_ctrl)
    
    results.append({
        "perturbation": pert,
        "edistance": edist,
        "n_cells": n_pert
    })

# Create dataframe, sort descending by E-distance, and save all perturbations to CSV
df_edist = pd.DataFrame(results).sort_values(by="edistance", ascending=False)
csv_out = OUT_DIR / f"edistance_full_genes_all_perturbations_{DATASET}.csv"
df_edist.to_csv(csv_out, index=False)
print(f"Saved sorted E-distance table with all perturbations to: {csv_out}")

print("Done!")