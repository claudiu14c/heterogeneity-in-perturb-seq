import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

print("Start")

DATASET = "replogle22"
PERTURBATIONS = ['PAFAH1B1', 'SEM1', 'RPA3', 'NOP9']
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

# Create masks for controls and specified perturbations
ctrl_mask = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
pert_mask = adata_full.obs["perturbation"].isin(PERTURBATIONS).values

n_ctrl = ctrl_mask.sum()
n_pert_total = pert_mask.sum()

print(f"Number of control cells available: {n_ctrl}")
print(f"Total perturbation cells across {PERTURBATIONS}: {n_pert_total}")

# --- 1. Fit PCA on Control Cells ONLY ---
adata_ctrl = adata_full[ctrl_mask].copy()
print("Running PCA on control cells only (n_comps=50)...")

X_ctrl = adata_ctrl.X.toarray() if hasattr(adata_ctrl.X, "toarray") else adata_ctrl.X
pca_model = PCA(n_components=50, random_state=42)
X_pca_ctrl = pca_model.fit_transform(X_ctrl)

# Store PCA results in control anndata object
adata_ctrl.obsm["X_pca"] = X_pca_ctrl
adata_ctrl.uns["pca"] = {
    "params": {"zero_center": True, "use_highly_variable": False, "n_pcs": 50, "svd_solver": "full"},
    "variance": pca_model.explained_variance_,
    "variance_ratio": pca_model.explained_variance_ratio_,
}
var_ratio_ctrl = pca_model.explained_variance_ratio_

# --- 2. Prepare Combined Subset and Project onto Frozen Control PCA ---
combined_mask = ctrl_mask | pert_mask
adata_combined = adata_full[combined_mask].copy()

print("Projecting combined controls + perturbations onto frozen control PCA...")
X_combined = adata_combined.X.toarray() if hasattr(adata_combined.X, "toarray") else adata_combined.X
X_pca_combined = pca_model.transform(X_combined)

adata_combined.obsm["X_pca"] = X_pca_combined
adata_combined.uns["pca"] = adata_ctrl.uns["pca"]
pcs_combined = X_pca_combined

# --- 3. Run t-SNE on the First 12 Frozen PCs and Plot ---
print("Running t-SNE on the first 12 frozen PCs...")
sc.tl.tsne(adata_combined, n_pcs=12, random_state=42)
tsne_coords = adata_combined.obsm["X_tsne"]

fig, ax = plt.subplots(figsize=(9, 7))

# Plot controls
is_ctrl_comb = (adata_combined.obs["perturbation"] == CONTROL_LABEL).values
ax.scatter(
    tsne_coords[is_ctrl_comb, 0], tsne_coords[is_ctrl_comb, 1], 
    c="gray", s=6, alpha=0.2, 
    label=f"{CONTROL_LABEL} (n={n_ctrl})", rasterized=True
)

# Plot each perturbation with a distinct color
colors = ["blue", "green", "purple", "orange", "cyan", "magenta", "brown"]
for i, pert in enumerate(PERTURBATIONS):
    is_pert = (adata_combined.obs["perturbation"] == pert).values
    n_pert = is_pert.sum()
    color = colors[i % len(colors)]
    
    ax.scatter(
        tsne_coords[is_pert, 0], tsne_coords[is_pert, 1], 
        c=color, s=6, alpha=0.6, 
        label=f"{pert} (n={n_pert})", rasterized=True
    )

ax.set_title(f"t-SNE (First 12 Frozen PCs, Controls-fitted) | {DATASET}")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9, markerscale=2)

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
        c=color, s=2, alpha=0.6, rasterized=True
    )

ax_inset.set_title("PC1 vs PC2 (Frozen)", fontsize=8, fontweight='bold')
ax_inset.set_xlabel(f"PC1 ({var_ratio_ctrl[0]*100:.1f}%)", fontsize=7)
ax_inset.set_ylabel(f"PC2 ({var_ratio_ctrl[1]*100:.1f}%)", fontsize=7)
ax_inset.tick_params(labelsize=6)

fig.tight_layout()

tsne_out_path = OUT_DIR / f"tsne_12pcs_frozen_pca_multi_perts.png"
fig.savefig(tsne_out_path, dpi=200)
plt.close(fig)
print("Saved t-SNE plot:", tsne_out_path)

print("Done!")