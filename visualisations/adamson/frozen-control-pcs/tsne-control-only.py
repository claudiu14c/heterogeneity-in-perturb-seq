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

# 1. Compute Highly Variable Genes globally on ALL cells in the dataset
print("Computing Global Highly Variable Genes (HVGs) on all cells...")
sc.pp.highly_variable_genes(adata_full, n_top_genes=2000, flavor="seurat")
hvg_mask = adata_full.var["highly_variable"].values
print(f"Number of global HVGs selected: {hvg_mask.sum()}")
adata_full = adata_full[:, hvg_mask].copy()

# Create mask for control cells only
ctrl_mask = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
n_ctrl = ctrl_mask.sum()
print(f"Number of control cells available: {n_ctrl}")

# Subset dataset to control cells only
adata_ctrl = adata_full[ctrl_mask].copy()

print("Running PCA on control cells (n_comps=50)...")
sc.pp.pca(adata_ctrl, n_comps=50)
var_ratio_ctrl = adata_ctrl.uns["pca"]["variance_ratio"]
pcs_ctrl = adata_ctrl.obsm["X_pca"]

# --- Run t-SNE computed on the first 12 PCs with Top-Left PCA Inset ---
print("Running t-SNE on the first 12 PCs (Controls Only)...")
sc.tl.tsne(adata_ctrl, n_pcs=12, random_state=42)
tsne_coords = adata_ctrl.obsm["X_tsne"]

fig, ax = plt.subplots(figsize=(8, 6))

# Plot control cells
ax.scatter(
    tsne_coords[:, 0], tsne_coords[:, 1], 
    c="gray", s=6, alpha=0.5, 
    label=f"{CONTROL_LABEL} (n={n_ctrl})", rasterized=True
)

ax.set_title(f"t-SNE (First 12 PCs) - Controls Only | {DATASET}")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

# --- Add PCA (PC1 vs PC2) Subplot Inset in the Top Left Corner ---
ax_inset = ax.inset_axes([0.03, 0.62, 0.32, 0.32])

ax_inset.scatter(
    pcs_ctrl[:, 0], pcs_ctrl[:, 1], 
    c="gray", s=2, alpha=0.5, rasterized=True
)

ax_inset.set_title("PC1 vs PC2", fontsize=8, fontweight='bold')
ax_inset.set_xlabel(f"PC1 ({var_ratio_ctrl[0]*100:.1f}%)", fontsize=7)
ax_inset.set_ylabel(f"PC2 ({var_ratio_ctrl[1]*100:.1f}%)", fontsize=7)
ax_inset.tick_params(labelsize=6)

fig.tight_layout()

tsne_out_path = OUT_DIR / f"tsne_12pcs_controls_only.png"
fig.savefig(tsne_out_path, dpi=200)
plt.close(fig)
print("Saved t-SNE plot:", tsne_out_path)

print("Done!")