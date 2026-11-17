
import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

print("Start")

DATASET = "replogle22"
PERTURBATIONS = ['PAFAH1B1', 'SEM1', 'RPA3', 'NOP9']
CONTROL_LABEL = "control"

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"../../../data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
adata_full = ad.read_h5ad(DATA_PATH)
print(f"Full dataset shape: {adata_full.shape}")

# --- 1. Compute global HVGs on the entire dataset ---
print("Computing global HVGs on all cells...")
sc.pp.highly_variable_genes(
    adata_full,
    n_top_genes=2000,
    flavor="seurat"
)

hvg_mask = adata_full.var["highly_variable"].values
print(f"Number of global HVGs selected: {hvg_mask.sum()}")

adata_full = adata_full[:, hvg_mask].copy()

# --- 2. Fit PCA on the ENTIRE dataset ---
print("Running PCA on the entire dataset (n_comps=50)...")

X_full = (
    adata_full.X.toarray()
    if hasattr(adata_full.X, "toarray")
    else adata_full.X
)

pca_model = PCA(n_components=50, random_state=42)
X_pca_full = pca_model.fit_transform(X_full)

adata_full.obsm["X_pca"] = X_pca_full
adata_full.uns["pca"] = {
    "params": {
        "zero_center": True,
        "use_highly_variable": False,
        "n_pcs": 50,
        "svd_solver": "full",
    },
    "variance": pca_model.explained_variance_,
    "variance_ratio": pca_model.explained_variance_ratio_,
}

var_ratio = pca_model.explained_variance_ratio_

print("PCA completed on the entire dataset.")

# --- 3. Select controls and specified perturbations ---
ctrl_mask = (
    adata_full.obs["perturbation"] == CONTROL_LABEL
).values

pert_mask = (
    adata_full.obs["perturbation"].isin(PERTURBATIONS)
).values

combined_mask = ctrl_mask | pert_mask

adata_combined = adata_full[combined_mask].copy()

n_ctrl = ctrl_mask.sum()
n_pert_total = pert_mask.sum()

print(f"Number of control cells available: {n_ctrl}")
print(f"Total perturbation cells across {PERTURBATIONS}: {n_pert_total}")
print(f"Combined subset shape: {adata_combined.shape}")

# Retrieve the already-computed PCA coordinates.
# No additional PCA fitting or projection is performed.
pcs_combined = adata_combined.obsm["X_pca"]

# --- 4. Run t-SNE on the first 12 PCs ---
print("Running t-SNE on the first 12 PCs...")

sc.tl.tsne(
    adata_combined,
    use_rep="X_pca",
    n_pcs=12,
    random_state=42
)

tsne_coords = adata_combined.obsm["X_tsne"]

# --- 5. Plot t-SNE ---
fig, ax = plt.subplots(figsize=(9, 7))

is_ctrl_comb = (
    adata_combined.obs["perturbation"] == CONTROL_LABEL
).values

# Plot controls
ax.scatter(
    tsne_coords[is_ctrl_comb, 0],
    tsne_coords[is_ctrl_comb, 1],
    c="gray",
    s=6,
    alpha=0.2,
    label=f"{CONTROL_LABEL} (n={n_ctrl})",
    rasterized=True
)

# Plot each perturbation
colors = ["blue", "green", "purple", "orange", "cyan", "magenta", "brown"]

for i, pert in enumerate(PERTURBATIONS):
    is_pert = (
        adata_combined.obs["perturbation"] == pert
    ).values

    n_pert = is_pert.sum()
    color = colors[i % len(colors)]

    ax.scatter(
        tsne_coords[is_pert, 0],
        tsne_coords[is_pert, 1],
        c=color,
        s=6,
        alpha=0.6,
        label=f"{pert} (n={n_pert})",
        rasterized=True
    )

ax.set_title(f"t-SNE (First 12 Global PCs) | {DATASET}")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(
    loc="upper right",
    framealpha=0.9,
    fontsize=9,
    markerscale=2
)

# --- 6. Add PC1 vs PC2 inset ---
ax_inset = ax.inset_axes([0.03, 0.62, 0.32, 0.32])

ax_inset.scatter(
    pcs_combined[is_ctrl_comb, 0],
    pcs_combined[is_ctrl_comb, 1],
    c="gray",
    s=2,
    alpha=0.2,
    rasterized=True
)

for i, pert in enumerate(PERTURBATIONS):
    is_pert = (
        adata_combined.obs["perturbation"] == pert
    ).values

    color = colors[i % len(colors)]

    ax_inset.scatter(
        pcs_combined[is_pert, 0],
        pcs_combined[is_pert, 1],
        c=color,
        s=2,
        alpha=0.6,
        rasterized=True
    )

ax_inset.set_title("PC1 vs PC2 (Global PCA)", fontsize=8, fontweight="bold")
ax_inset.set_xlabel(
    f"PC1 ({var_ratio[0]*100:.1f}%)",
    fontsize=7
)
ax_inset.set_ylabel(
    f"PC2 ({var_ratio[1]*100:.1f}%)",
    fontsize=7
)
ax_inset.tick_params(labelsize=6)

fig.tight_layout()

tsne_out_path = OUT_DIR / "tsne_12pcs_global_pca_multi_perts_hvg.png"
fig.savefig(tsne_out_path, dpi=200)
plt.close(fig)

print("Saved t-SNE plot:", tsne_out_path)
print("Done!")


# --- 2b. Plot PCA explained variance (Elbow plot) ---
print("Plotting PCA explained variance...")

pcs = np.arange(1, 51)
individual_variance = var_ratio * 100
cumulative_variance = np.cumsum(var_ratio) * 100

fig, ax1 = plt.subplots(figsize=(9, 6))

# Individual explained variance
ax1.plot(
    pcs,
    individual_variance,
    marker="o",
    markersize=4,
    linewidth=1.5,
    color="steelblue",
    label="Individual explained variance"
)

ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Explained Variance (%)", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.set_xticks(np.arange(0, 51, 5))
ax1.grid(alpha=0.25)

# Cumulative explained variance
ax2 = ax1.twinx()
ax2.plot(
    pcs,
    cumulative_variance,
    marker=".",
    markersize=3,
    linewidth=1.5,
    color="darkorange",
    label="Cumulative explained variance"
)

ax2.set_ylabel("Cumulative Explained Variance (%)", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")
ax2.set_ylim(0, 105)

# Combined legend
lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="center right", framealpha=0.9)

ax1.set_title(f"PCA Explained Variance (First 50 PCs) | {DATASET}")

fig.tight_layout()

pca_out_path = OUT_DIR / "pca_explained_variance_global.png"
fig.savefig(pca_out_path, dpi=200)
plt.close(fig)

print("Saved PCA elbow plot:", pca_out_path)