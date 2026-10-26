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

# Automatically get all unique perturbations excluding the control label
all_perturbations = adata_full.obs["perturbation"].astype(str).unique()
PERTURBATIONS = [p for p in all_perturbations if p != CONTROL_LABEL]
print(f"Found {len(PERTURBATIONS)} perturbations in total.")

# 1. Compute Highly Variable Genes (HVGs) on the normalized dataset
print("Computing Highly Variable Genes (HVGs)...")
sc.pp.highly_variable_genes(adata_full, n_top_genes=2000, flavor="seurat")
hvg_mask = adata_full.var["highly_variable"].values
print(f"Number of HVGs selected: {hvg_mask.sum()}")

# 2. Prepare Combined Subset using HVGs
pert_mask_all = adata_full.obs["perturbation"].isin(PERTURBATIONS).values
ctrl_mask = (adata_full.obs["perturbation"] == CONTROL_LABEL).values

pert_indices = np.where(pert_mask_all)[0]
ctrl_indices = np.where(ctrl_mask)[0]
combined_indices = np.concatenate([pert_indices, ctrl_indices])

adata_combined = adata_full[combined_indices, hvg_mask].copy()

# 3. Run PCA (12 components) and t-SNE on HVGs
print("Running PCA (12 components) on HVGs...")
sc.pp.pca(adata_combined, n_comps=12, random_state=42)
pcs_combined = adata_combined.obsm["X_pca"]
var_ratio_combined = adata_combined.uns["pca"]["variance_ratio"]

print("Running t-SNE on the first 12 PCs...")
sc.tl.tsne(adata_combined, n_pcs=12, random_state=42)
tsne_coords = adata_combined.obsm["X_tsne"]

# 4. Plotting Single Panel: t-SNE (on 12 PCs) with PCA inset (Binary coloring)
fig, ax = plt.subplots(figsize=(8, 6))

is_ctrl_comb = (adata_combined.obs["perturbation"] == CONTROL_LABEL).values
is_pert_comb = ~is_ctrl_comb

# Plot controls first (gray)
ax.scatter(
    tsne_coords[is_ctrl_comb, 0], tsne_coords[is_ctrl_comb, 1], 
   
    c="#0072B2", s=0.7, alpha=0.6,
    label=f"Control (n={is_ctrl_comb.sum()})", rasterized=True
)

# Plot all perturbations combined (single color)
ax.scatter(
    tsne_coords[is_pert_comb, 0], tsne_coords[is_pert_comb, 1], 
    c="gray", s=0.7, alpha=0.2,
    label=f"Perturbed (n={is_pert_comb.sum()})", rasterized=True
)

ax.set_title("t-SNE (on first 12 PCs, HVGs - Binary View)", fontsize=11, fontweight='bold')
ax.set_xlabel("t-SNE 1", fontsize=10)
ax.set_ylabel("t-SNE 2", fontsize=10)
ax.legend(loc="upper right", framealpha=0.9, fontsize=9, markerscale=1.5)

# Subplot inset in the top left corner (PC1 vs PC2)
ax_inset = ax.inset_axes([0.03, 0.62, 0.32, 0.32])

ax_inset.scatter(
    pcs_combined[is_ctrl_comb, 0], pcs_combined[is_ctrl_comb, 1], 
    c="#0072B2", s=0.1, alpha=0.4, rasterized=True
)
ax_inset.scatter(
    pcs_combined[is_pert_comb, 0], pcs_combined[is_pert_comb, 1], 
    c="gray", s=0.1, alpha=0.1, rasterized=True
)

ax_inset.set_title("PC1 vs PC2", fontsize=8, fontweight='bold')
ax_inset.set_xlabel(f"PC1 ({var_ratio_combined[0]*100:.1f}%)", fontsize=7)
ax_inset.set_ylabel(f"PC2 ({var_ratio_combined[1]*100:.1f}%)", fontsize=7)
ax_inset.tick_params(labelsize=6)

fig.suptitle(f"Dataset: {DATASET} | Binary Perturbation View (Top 2,000 HVGs)", fontsize=10)
fig.tight_layout()

out_path = OUT_DIR / f"tsne12_inset_pca_{DATASET}_binary_hvgs.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print("Saved:", out_path)

print("Done!")