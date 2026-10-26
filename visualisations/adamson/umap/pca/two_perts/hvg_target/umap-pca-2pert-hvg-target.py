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

# 1. Subset dataset to control + the 2 chosen perturbations only
target_groups = PERTURBATIONS + [CONTROL_LABEL]
subset_mask = adata_full.obs["perturbation"].isin(target_groups).values
adata_sub = adata_full[subset_mask, :].copy()
print(f"Subset shape (Control + {PERTURBATIONS}): {adata_sub.shape}")

# 2. Compute HVGs exclusively on this subset
print("Computing Highly Variable Genes (HVGs) on the subset...")
sc.pp.highly_variable_genes(adata_sub, n_top_genes=2000, flavor="seurat")
hvg_mask = adata_sub.var["highly_variable"].values
print(f"Number of subset-specific HVGs selected: {hvg_mask.sum()}")

# Keep only the HVGs
adata_sub = adata_sub[:, hvg_mask].copy()

# 3. Run PCA (12 components) and t-SNE on the subset HVGs
print("Running PCA (12 components) on subset HVGs...")
sc.pp.pca(adata_sub, n_comps=12, random_state=42)
pcs_sub = adata_sub.obsm["X_pca"]
var_ratio_sub = adata_sub.uns["pca"]["variance_ratio"]

print("Running UMAP on the first 12 PCs...")
sc.pp.neighbors(adata_sub, n_pcs=12, random_state=42)
sc.tl.umap(adata_sub, random_state=42)
umap_coords = adata_sub.obsm["X_umap"]

# 4. Plotting: UMAP with PCA inset (Small points for large cell counts)
fig, ax = plt.subplots(figsize=(8, 6))

is_ctrl = (adata_sub.obs["perturbation"] == CONTROL_LABEL).values
colors = ["#0072B2", "#D55E00"]  # Distinct colors for the 2 perturbations

# Plot control cells (gray)
ax.scatter(
    umap_coords[is_ctrl, 0], umap_coords[is_ctrl, 1], 
    c="gray", s=5.0, alpha=0.3, 
    label=f"Control (n={is_ctrl.sum()})", rasterized=True
)

# Plot each of the 2 perturbations
for i, pert in enumerate(PERTURBATIONS):
    is_pert = (adata_sub.obs["perturbation"] == pert).values
    ax.scatter(
        umap_coords[is_pert, 0], umap_coords[is_pert, 1], 
        c=colors[i], s=5.0, alpha=0.4, 
        label=f"{pert} (n={is_pert.sum()})", rasterized=True
    )

ax.set_title(f"UMAP (on 12 PCs, Subset HVGs: {PERTURBATIONS[0]} & {PERTURBATIONS[1]})", fontsize=10, fontweight='bold')
ax.set_xlabel("UMAP 1", fontsize=10)
ax.set_ylabel("UMAP 2", fontsize=10)
ax.legend(loc="upper right", framealpha=0.9, fontsize=9, markerscale=3.0)

# Subplot inset in the top left corner (PC1 vs PC2)
ax_inset = ax.inset_axes([0.03, 0.62, 0.32, 0.32])

ax_inset.scatter(
    pcs_sub[is_ctrl, 0], pcs_sub[is_ctrl, 1], 
    c="gray", s=0.3, alpha=0.3, rasterized=True
)
for i, pert in enumerate(PERTURBATIONS):
    is_pert = (adata_sub.obs["perturbation"] == pert).values
    ax_inset.scatter(
        pcs_sub[is_pert, 0], pcs_sub[is_pert, 1], 
        c=colors[i], s=0.3, alpha=0.4, rasterized=True
    )

ax_inset.set_title("PC1 vs PC2", fontsize=8, fontweight='bold')
ax_inset.set_xlabel(f"PC1 ({var_ratio_sub[0]*100:.1f}%)", fontsize=7)
ax_inset.set_ylabel(f"PC2 ({var_ratio_sub[1]*100:.1f}%)", fontsize=7)
ax_inset.tick_params(labelsize=6)

fig.suptitle(f"Dataset: {DATASET} | Focused Subset Analysis", fontsize=10)
fig.tight_layout()

out_path = OUT_DIR / f"umap12_pca_{DATASET}_subset_hvgs_{PERTURBATIONS[0]}_{PERTURBATIONS[1]}.png"
fig.savefig(out_path, dpi=300)
plt.close(fig)
print("Saved:", out_path)

print("Done!")