import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ks_2samp
from matplotlib.lines import Line2D

print("Start")

DATASET = "adamson16"
PERTURBATIONS = ["AMIGO3", "ARHGAP22"]
CONTROL_LABEL = "control"  # Change to "control" or "non-targeting" based on dataset
KS_THRESHOLD = 0.04  # Threshold on the KS statistic D to select differentially expressed genes

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

    # Get indices & randomly downsample controls to match perturbation cells
    pert_indices = np.where(pert_mask)[0]
    ctrl_indices = np.where(ctrl_mask)[0]

    np.random.seed(42)  # Set seed for reproducibility
    if n_ctrl > n_pert:
        sampled_ctrl_indices = np.random.choice(ctrl_indices, size=n_pert, replace=False)
    else:
        sampled_ctrl_indices = ctrl_indices

    # Combine indices and subset AnnData
    combined_indices = np.concatenate([pert_indices, sampled_ctrl_indices])
    adata = adata_full[combined_indices].copy()
    print(f"Balanced subset shape (pert + sampled control): {adata.shape}")

    # --- 1. Find Differentially Expressed Genes Before PCA (using KS test) ---
    X_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    gene_names = adata.var_names
    
    pert_mask_sub = (adata.obs["perturbation"] == PERTURBATION).values
    ctrl_mask_sub = (adata.obs["perturbation"] == CONTROL_LABEL).values

    ks_results = []
    for i, gene in enumerate(gene_names):
        expr_pert = X_data[pert_mask_sub, i]
        expr_ctrl = X_data[ctrl_mask_sub, i]
        res = ks_2samp(expr_pert, expr_ctrl)
        ks_results.append({
            'gene': gene,
            'statistic_D': res.statistic,
            'pvals': res.pvalue
        })

    top_genes_df = pd.DataFrame(ks_results)
    top_genes_df = top_genes_df.sort_values(by='statistic_D', ascending=False).reset_index(drop=True)

    # Save DE table to CSV
    de_csv_path = OUT_DIR / f"de_genes_{DATASET}_{PERTURBATION}.csv"
    top_genes_df.to_csv(de_csv_path, index=False)
    print(f"Saved DE table: {de_csv_path}")

    # --- 2. Subset AnnData to DE Genes passing KS threshold for PCA ---
    selected_genes = top_genes_df.loc[top_genes_df['statistic_D'] >= KS_THRESHOLD, 'gene'].values
    # Ensure genes exist in adata variables
    selected_genes = [g for g in selected_genes if g in adata.var_names]
    
    print(f"Running PCA on {len(selected_genes)} DE genes passing KS threshold D >= {KS_THRESHOLD}...")
    if len(selected_genes) == 0:
        print("Warning: No genes passed the KS threshold. Falling back to top 500 genes.")
        selected_genes = top_genes_df['gene'].head(500).values
        selected_genes = [g for g in selected_genes if g in adata.var_names]

    adata_pca = adata[:, selected_genes].copy()

    n_comps = min(50, len(selected_genes), adata_pca.n_obs - 1)
    sc.pp.pca(adata_pca, n_comps=n_comps)
    pcs = adata_pca.obsm["X_pca"]
    var_ratio = adata_pca.uns["pca"]["variance_ratio"]

    print(f"Total variance explained by {n_comps} PCs: {var_ratio.sum() * 100:.2f}%")

    # Separate PC coordinates by group
    is_pert = (adata_pca.obs["perturbation"] == PERTURBATION).values
    pcs_pert = pcs[is_pert]
    pcs_ctrl = pcs[~is_pert]

    # Compute means (centroids) for PC1 and PC2
    mean_pert = np.mean(pcs_pert[:, :2], axis=0)
    mean_ctrl = np.mean(pcs_ctrl[:, :2], axis=0)

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Scatter plot
    ax.scatter(pcs_pert[:, 0], pcs_pert[:, 1], c="blue", s=6, alpha=0.4, label=f"{PERTURBATION} (n={n_pert})", rasterized=True)
    ax.scatter(pcs_ctrl[:, 0], pcs_ctrl[:, 1], c="red", s=6, alpha=0.4, label=f"{CONTROL_LABEL} (n={len(pcs_ctrl)})", rasterized=True)

    # Plot means / centroids
    ax.scatter(mean_pert[0], mean_pert[1], c="blue", marker="*", s=250, edgecolor="black", linewidth=1.2, label=f"{PERTURBATION} Mean", zorder=5)
    ax.scatter(mean_ctrl[0], mean_ctrl[1], c="red", marker="*", s=250, edgecolor="black", linewidth=1.2, label=f"{CONTROL_LABEL} Mean", zorder=5)

    # Fit and plot Linear Regression Lines (PC2 vs PC1)
    slope_pert, intercept_pert = np.polyfit(pcs_pert[:, 0], pcs_pert[:, 1], 1)
    x_pert_range = np.linspace(pcs_pert[:, 0].min(), pcs_pert[:, 0].max(), 100)
    ax.plot(x_pert_range, slope_pert * x_pert_range + intercept_pert, color="darkblue", linewidth=2.5, linestyle="--", label=f"{PERTURBATION} Fit")

    slope_ctrl, intercept_ctrl = np.polyfit(pcs_ctrl[:, 0], pcs_ctrl[:, 1], 1)
    x_ctrl_range = np.linspace(pcs_ctrl[:, 0].min(), pcs_ctrl[:, 0].max(), 100)
    ax.plot(x_ctrl_range, slope_ctrl * x_ctrl_range + intercept_ctrl, color="darkred", linewidth=2.5, linestyle="--", label=f"{CONTROL_LABEL} Fit")

    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_title(f"PCA (KS DE, D >= {KS_THRESHOLD}): {DATASET}\n{PERTURBATION} vs Control")
    
    fig.tight_layout()

    out_path = OUT_DIR / f"pca_deg_{DATASET}_{PERTURBATION}_vs_balanced_control.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)

print("Done!")