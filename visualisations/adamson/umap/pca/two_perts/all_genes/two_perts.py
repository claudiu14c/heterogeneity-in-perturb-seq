import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

print("Start")

DATASET = "adamson16"
PERTURBATIONS = ["STT3A", "OST4"]
CONTROL_LABEL = "control"  # Change to "control" or "non-targeting" based on dataset

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
adata_full = ad.read_h5ad(DATA_PATH)
print(f"Full dataset shape: {adata_full.shape}")

# Create masks for controls and all specified perturbations
ctrl_mask = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
pert_mask = adata_full.obs["perturbation"].isin(PERTURBATIONS).values

n_ctrl = ctrl_mask.sum()
n_pert_total = pert_mask.sum()

print(f"Number of control cells available: {n_ctrl}")
print(f"Total perturbation cells across {PERTURBATIONS}: {n_pert_total}")

ctrl_indices = np.where(ctrl_mask)[0]
pert_indices = np.where(pert_mask)[0]

# Prepare Combined subset (Control + All Perturbations)
combined_indices = np.concatenate([pert_indices, ctrl_indices])
adata_combined = adata_full[combined_indices].copy()

print("Running PCA on combined controls + all perturbations (n_comps=50)...")
sc.pp.pca(adata_combined, n_comps=50)
var_ratio_combined = adata_combined.uns["pca"]["variance_ratio"]
pcs_combined = adata_combined.obsm["X_pca"]

# --- 1. Plot Variance Explained for the first 50 PCs ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    range(1, 51), var_ratio_combined[:50] * 100, 
    marker="o", linestyle="-", color="b", markersize=4, rasterized=True
)
ax.set_title(f"PCA Variance Explained | Dataset: {DATASET}")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained (%)")
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()

var_out_path = OUT_DIR / f"pca_variance_ratio_two_perts.png"
fig.savefig(var_out_path, dpi=200)
plt.close(fig)
print("Saved variance plot:", var_out_path)

# --- 2. Run and Plot UMAP computed on the first 12 PCs with Top-Left PCA Inset ---
print("Running UMAP on the first 12 PCs...")
sc.pp.neighbors(adata_combined, n_pcs=12, random_state=42)
sc.tl.umap(adata_combined, random_state=42)
umap_coords = adata_combined.obsm["X_umap"]

fig, ax = plt.subplots(figsize=(8, 6))

# Plot controls
is_ctrl_comb = (adata_combined.obs["perturbation"] == CONTROL_LABEL).values
ax.scatter(
    umap_coords[is_ctrl_comb, 0], umap_coords[is_ctrl_comb, 1], 
    c="gray", s=6, alpha=0.2, 
    label=f"{CONTROL_LABEL} (n={n_ctrl})", rasterized=True
)

# Plot each perturbation with a distinct color
colors = ["blue", "green", "purple", "orange", "cyan"]
for i, pert in enumerate(PERTURBATIONS):
    is_pert = (adata_combined.obs["perturbation"] == pert).values
    n_pert = is_pert.sum()
    color = colors[i % len(colors)]
    
    ax.scatter(
        umap_coords[is_pert, 0], umap_coords[is_pert, 1], 
        c=color, s=6, alpha=0.5, 
        label=f"{pert} (n={n_pert})", rasterized=True
    )

ax.set_title(f"UMAP (First 12 PCs) | {DATASET}")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

# --- Add PCA (PC1 vs PC2) Subplot Inset in the Top Left Corner ---
ax_inset = ax.inset_axes([0.03, 0.62, 0.32, 0.32])

ax_inset.scatter(
    pcs_combined[is_ctrl_comb, 0], pcs_combined[is_ctrl_comb, 1], 
    c="gray", s=2, alpha=0.2, rasterized=True
)
for i, pert in enumerate(PERTURBATIONS):
    is_pert = (adata_combined.obs["perturbation"] == pert).values
    color = colors[i % len(colors)]
    ax_inset.scatter(
        pcs_combined[is_pert, 0], pcs_combined[is_pert, 1], 
        c=color, s=2, alpha=0.5, rasterized=True
    )

ax_inset.set_title("PC1 vs PC2", fontsize=8, fontweight='bold')
ax_inset.set_xlabel(f"PC1 ({var_ratio_combined[0]*100:.1f}%)", fontsize=7)
ax_inset.set_ylabel(f"PC2 ({var_ratio_combined[1]*100:.1f}%)", fontsize=7)
ax_inset.tick_params(labelsize=6)

fig.tight_layout()

umap_out_path = OUT_DIR / f"umap_12pcs_two_perts.png"
fig.savefig(umap_out_path, dpi=200)
plt.close(fig)
print("Saved UMAP plot:", umap_out_path)

print("Done!")