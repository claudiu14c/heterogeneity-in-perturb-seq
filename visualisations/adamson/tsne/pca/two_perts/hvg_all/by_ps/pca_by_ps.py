import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import Ridge

print("Start t-SNE colored by PS Score (Separate plots per cell type)")

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

# 1. Compute Global Highly Variable Genes (HVGs) on all cells in the dataset
print("Computing Global Highly Variable Genes (HVGs) on all cells...")
sc.pp.highly_variable_genes(adata_full, n_top_genes=2000, flavor="seurat")
hvg_mask = adata_full.var["highly_variable"].values

ctrl_mask_full = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
ctrl_indices = np.where(ctrl_mask_full)[0]

# 2. Process each perturbation independently against control
for pert in PERTURBATIONS:
    print(f"\nProcessing perturbation: {pert}")
    pert_mask = (adata_full.obs["perturbation"] == pert).values
    pert_indices = np.where(pert_mask)[0]

    # Subset dataset to this perturbation + control, keeping only global HVGs
    sub_indices = np.concatenate([pert_indices, ctrl_indices])
    adata_sub = adata_full[sub_indices, hvg_mask].copy()

    sub_pert_mask = (adata_sub.obs["perturbation"] == pert).values
    sub_ctrl_mask = (adata_sub.obs["perturbation"] == CONTROL_LABEL).values

    # Step 1: Identify downstream DEGs using Wilcoxon rank-sum test
    sc.tl.rank_genes_groups(
        adata_sub, 
        groupby="perturbation", 
        groups=[pert], 
        reference=CONTROL_LABEL, 
        method="wilcoxon"
    )
    de_df = sc.get.rank_genes_groups_df(adata_sub, group=pert)
    
    de_df = de_df.sort_values(by="pvals", ascending=True)
    filtered_de = de_df[(de_df["logfoldchanges"].abs() > 0.1)]
    if len(filtered_de) < 10:
        filtered_de = de_df.head(min(10, len(de_df)))
    
    target_genes = filtered_de["names"].head(50).values
    if len(target_genes) < 5:
        target_genes = de_df["names"].head(20).values

    # Step 2: Compute PS Score via Ridge Regression beta weights
    X_data = adata_sub.X.toarray() if hasattr(adata_sub.X, "toarray") else adata_sub.X
    gene_names = adata_sub.var_names

    valid_target_genes = [g for g in target_genes if g in gene_names]
    if len(valid_target_genes) == 0:
        valid_target_genes = gene_names[:20]

    gene_indices = [np.where(gene_names == g)[0][0] for g in valid_target_genes]
    Y = X_data[:, gene_indices]
    D = sub_pert_mask.astype(int).reshape(-1, 1)

    ridge = Ridge(alpha=0.01)
    ridge.fit(D, Y)
    beta_weights = ridge.coef_[:, 0]

    Y_control_mean = Y[sub_ctrl_mask].mean(axis=0)
    Y_centered = Y - Y_control_mean

    ps_scores = np.dot(Y_centered, beta_weights)

    # Min-Max scale PS scores between 0 and 1
    ps_min, ps_max = ps_scores.min(), ps_scores.max()
    ps_scores_normalized = (ps_scores - ps_min) / (ps_max - ps_min + 1e-8)
    adata_sub.obs["ps_score"] = ps_scores_normalized

    # Step 3: Run PCA and t-SNE for this subset
    sc.pp.pca(adata_sub, n_comps=12, random_state=42)
    sc.tl.tsne(adata_sub, n_pcs=12, random_state=42)
    tsne_coords = adata_sub.obsm["X_tsne"]

    # --- PLOT 1: Perturbation Only ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sc_plot = ax.scatter(
        tsne_coords[sub_pert_mask, 0], tsne_coords[sub_pert_mask, 1],
        c=adata_sub.obs.loc[sub_pert_mask, "ps_score"].values, cmap="coolwarm",
        vmin=0.0, vmax=1.0,
        s=5.0, alpha=0.7, rasterized=True
    )
    ax.set_title(f"t-SNE: {pert} cells only (PS Score)", fontsize=10, fontweight='bold')
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    fig.colorbar(sc_plot, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout()
    out_path_pert = OUT_DIR / f"tsne_ps_score_{pert}_only_{DATASET}.png"
    fig.savefig(out_path_pert, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path_pert}")

    # --- PLOT 2: Control Cells Only (relative to this perturbation's signature) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sc_plot = ax.scatter(
        tsne_coords[sub_ctrl_mask, 0], tsne_coords[sub_ctrl_mask, 1],
        c=adata_sub.obs.loc[sub_ctrl_mask, "ps_score"].values, cmap="coolwarm",
        vmin=0.0, vmax=1.0,
        s=5.0, alpha=0.7, rasterized=True
    )
    ax.set_title(f"t-SNE: Control cells only (PS Score vs {pert})", fontsize=10, fontweight='bold')
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    fig.colorbar(sc_plot, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout()
    out_path_ctrl = OUT_DIR / f"tsne_ps_score_control_for_{pert}_{DATASET}.png"
    fig.savefig(out_path_ctrl, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path_ctrl}")

print("Done processing all plots!")