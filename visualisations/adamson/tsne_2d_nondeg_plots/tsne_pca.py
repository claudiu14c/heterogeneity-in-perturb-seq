import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.lines import Line2D
from scipy.stats import ranksums

print("Start")

DATASET = "adamson16"
PERTURBATIONS = ["AMIGO3", "ARHGAP22"]
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

for PERTURBATION in PERTURBATIONS:
    print(f"\n--- Processing perturbation: {PERTURBATION} ---")

    # Create masks for perturbation and control
    pert_mask = (adata_full.obs["perturbation"] == PERTURBATION).values
    ctrl_mask = (adata_full.obs["perturbation"] == CONTROL_LABEL).values

    n_pert = pert_mask.sum()
    n_ctrl = ctrl_mask.sum()

    print(f"Number of perturbation cells: {n_pert}")
    print(f"Number of control cells available: {n_ctrl}")

    if n_pert == 0:
        print(f"Warning: No cells found for {PERTURBATION}. Skipping.")
        continue

    # Get indices
    pert_indices = np.where(pert_mask)[0]
    ctrl_indices = np.where(ctrl_mask)[0]

    # Randomly downsample controls to match the number of perturbation cells
    np.random.seed(42)  # Set seed for reproducibility
    if n_ctrl > n_pert:
        sampled_ctrl_indices = np.random.choice(ctrl_indices, size=n_pert, replace=False)
    else:
        sampled_ctrl_indices = ctrl_indices

    # Combine indices and subset AnnData
    combined_indices = np.concatenate([pert_indices, sampled_ctrl_indices])
    adata = adata_full[combined_indices].copy()
    print(f"Balanced subset shape (pert + sampled control): {adata.shape}")

    # PCA first (standard pre-requisite for t-SNE in scanpy)
    n_comps = 50
    sc.pp.pca(adata, n_comps=n_comps)
    var_ratio = adata.uns["pca"]["variance_ratio"]

    print(f"Total variance explained by {n_comps} PCs: {var_ratio.sum() * 100:.2f}%")

    # --- 1. Save Scree Plot as a Separate File ---
    fig_scree, ax_scree = plt.subplots(figsize=(7, 4.5))
    ax_scree.plot(range(1, len(var_ratio) + 1), var_ratio * 100, marker='o', linestyle='-', color='teal', markersize=4)
    ax_scree.set_xlabel("Principal Component")
    ax_scree.set_ylabel("Variance Explained (%)")
    ax_scree.set_title(f"Scree Plot: {DATASET} - {PERTURBATION}")
    fig_scree.tight_layout()
    
    scree_path = OUT_DIR / f"scree_plot_pca_{DATASET}_{PERTURBATION}.png"
    fig_scree.savefig(scree_path, dpi=200)
    plt.close(fig_scree)
    print("Saved scre:", scree_path)

    # --- 2. Run t-SNE on PCA representation ---
    print("Running t-SNE...")
    sc.tl.tsne(adata, n_pcs=12, random_state=42)
    tsne_coords = adata.obsm["X_tsne"]

    # Separate t-SNE coordinates by group
    is_pert = (adata.obs["perturbation"] == PERTURBATION).values
    tsne_pert = tsne_coords[is_pert]
    tsne_ctrl = tsne_coords[~is_pert]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Scatter plot
    ax.scatter(tsne_pert[:, 0], tsne_pert[:, 1], c="blue", s=6, alpha=0.4, label=f"{PERTURBATION} (n={n_pert})", rasterized=True)
    ax.scatter(tsne_ctrl[:, 0], tsne_ctrl[:, 1], c="red", s=6, alpha=0.4, label=f"{CONTROL_LABEL} (n={len(tsne_ctrl)})", rasterized=True)

    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    # Remove axis ticks and labels for clean t-SNE visualization
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_title(f"t-SNE: {DATASET}\n{PERTURBATION} vs Balanced Control")
    
    fig.tight_layout()

    out_path = OUT_DIR / f"tsne_{DATASET}_{PERTURBATION}_vs_balanced_control.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved t-SNE plot:", out_path)

print("Done!")