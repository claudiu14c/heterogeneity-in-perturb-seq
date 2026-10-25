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
# unreliable, reliable
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

# 1. Prepare Controls-only subset for Subplot 1
adata_ctrl = adata_full[ctrl_indices].copy()
sc.pp.pca(adata_ctrl, n_comps=20)
pcs_ctrl_only = adata_ctrl.obsm["X_pca"]
var_ratio_ctrl = adata_ctrl.uns["pca"]["variance_ratio"]

# 2. Prepare Combined subset (Control + All Perturbations) for Subplot 2
combined_indices = np.concatenate([pert_indices, ctrl_indices])
adata_combined = adata_full[combined_indices].copy()

print("Running PCA on combined controls + all perturbations...")
sc.pp.pca(adata_combined, n_comps=50)
pcs_combined = adata_combined.obsm["X_pca"]
var_ratio_combined = adata_combined.uns["pca"]["variance_ratio"]

# Create figure with 2 subplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Subplot 1: Controls only ---
axes[0].scatter(
    pcs_ctrl_only[:, 0], pcs_ctrl_only[:, 1], 
    c="red", s=6, alpha=0.4, 
    label=f"{CONTROL_LABEL} (n={n_ctrl})", rasterized=True
)
axes[0].set_title("PCA: Controls Only")
axes[0].set_xlabel(f"PC1 ({var_ratio_ctrl[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({var_ratio_ctrl[1]*100:.1f}%)")
axes[0].legend(loc="upper right", framealpha=0.9, fontsize=9)

# --- Subplot 2: Controls + All Perturbations on the same plot ---
is_ctrl_comb = (adata_combined.obs["perturbation"] == CONTROL_LABEL).values
axes[1].scatter(
    pcs_combined[is_ctrl_comb, 0], pcs_combined[is_ctrl_comb, 1], 
    c="gray", s=6, alpha=0.2, 
    label=f"{CONTROL_LABEL} (n={n_ctrl})", rasterized=True
)

# Plot each perturbation with a distinct color
colors = ["blue", "green", "purple", "orange", "cyan"]
for i, pert in enumerate(PERTURBATIONS):
    is_pert = (adata_combined.obs["perturbation"] == pert).values
    n_pert = is_pert.sum()
    color = colors[i % len(colors)]
    
    axes[1].scatter(
        pcs_combined[is_pert, 0], pcs_combined[is_pert, 1], 
        c=color, s=6, alpha=0.5, 
        label=f"{pert} (n={n_pert})", rasterized=True
    )

axes[1].set_title(f"PCA: Combined Perturbations vs All Controls")
axes[1].set_xlabel(f"PC1 ({var_ratio_combined[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({var_ratio_combined[1]*100:.1f}%)")
axes[1].legend(loc="upper right", framealpha=0.9, fontsize=9)

fig.suptitle(f"Dataset: {DATASET} | Combined Perturbations", fontsize=12)
fig.tight_layout()

out_path = OUT_DIR / f"pca12_{DATASET}_combined_perturbations.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print("Saved:", out_path)

print("Done!")

# Stop subsampling
# Always compute the Highly variable genes and DEGS on the whole population

# 1. Why do we have clusters on tsne
#     Cell cycle package (ICA , cell cycle, gene expression, PS score)
#     DO DEGS, HVG, nothing
# 2. Why don t they correspond to splitting control from perturb?
#     Look at edistance per perturbation.
#     Look at the graph - compare it with Anna