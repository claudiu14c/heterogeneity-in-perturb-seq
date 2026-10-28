import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

print("Start t-SNE colored by Target Gene Expression (Blue-to-Red Colormap)")

DATASET = "adamson16"
PERTURBATIONS = ["STT3A", "OST4"]

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

# 2. Process each target perturbation individually (no control, only perturbation cells)
for pert in PERTURBATIONS:
    print(f"\nProcessing perturbation: {pert}")
    
    pert_mask = (adata_full.obs["perturbation"] == pert).values
    n_pert = pert_mask.sum()

    if n_pert == 0:
        print(f"Warning: No cells found for {pert}.")
        continue

    pert_indices = np.where(pert_mask)[0]

    # Subset data to this perturbation's cells and global HVGs
    adata_sub = adata_full[pert_indices, hvg_mask].copy()

    # Check if target gene exists in dataset for expression coloring
    if pert not in adata_full.var_names:
        print(f"Gene {pert} not found in dataset var_names.")
        continue

    # Extract target gene expression for these cells
    gene_expr = adata_full[pert_indices, pert].X
    if hasattr(gene_expr, "toarray"):
        gene_expr = gene_expr.toarray().flatten()
    else:
        gene_expr = gene_expr.flatten()

    adata_sub.obs["target_gene_expr"] = gene_expr

    # Step 3: Run PCA and t-SNE for this perturbation subset
    print(f"Running PCA and t-SNE for {pert}...")
    sc.pp.pca(adata_sub, n_comps=12, random_state=42)
    sc.tl.tsne(adata_sub, n_pcs=12, random_state=42)
    tsne_coords = adata_sub.obsm["X_tsne"]

    # Step 4: Plot t-SNE colored by target gene expression using coolwarm (blue to red)
    fig, ax = plt.subplots(figsize=(8, 6))
    expr_vals = adata_sub.obs["target_gene_expr"].values
    
    sc_plot = ax.scatter(
        tsne_coords[:, 0], tsne_coords[:, 1],
        c=expr_vals, cmap="coolwarm",
        vmin=expr_vals.min(), vmax=expr_vals.max(),
        s=5.0, alpha=0.7, rasterized=True
    )

    ax.set_title(f"t-SNE: {pert} cells only (Target Gene Expression)", fontsize=10, fontweight='bold')
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    
    fig.colorbar(sc_plot, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout()

    out_path = OUT_DIR / f"tsne_target_expr_{pert}_{DATASET}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

print("Done!")