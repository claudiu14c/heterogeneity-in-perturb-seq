import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

DATASET = "adamson16"
CONTROL_LABEL = "control"
THRESHOLDS = [2.0, 2.05, 2.1]

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

adata_full = ad.read_h5ad(DATA_PATH)
adata_ctrl = adata_full[adata_full.obs["perturbation"] == CONTROL_LABEL].copy()

# 1. Score the Cytoplasmic_Ribo_Mito gene set
ribo_mito_genes = [
    "MT-CO3", "MT-CYB", "MT-CO2", "MT-ND4", "RPL21", "RPS23",
    "RPL12", "RPS3A", "RPL7", "RPS6"
]
present_genes = [g for g in ribo_mito_genes if g in adata_ctrl.var_names]
sc.tl.score_genes(adata_ctrl, gene_list=present_genes, score_name="score_Cytoplasmic_Ribo_Mito")

scores = adata_ctrl.obs["score_Cytoplasmic_Ribo_Mito"].to_numpy()
print(f"Score summary -> min: {scores.min():.3f}, median: {np.median(scores):.3f}, max: {scores.max():.3f}")

# 2. Run PCA on controls
sc.pp.pca(adata_ctrl, n_comps=20)
pcs = adata_ctrl.obsm["X_pca"]
var_ratio = adata_ctrl.uns["pca"]["variance_ratio"]

# Consistent axis limits across all subplots so visual comparison is exact
xlim = (pcs[:, 0].min() - 1, pcs[:, 0].max() + 1)
ylim = (pcs[:, 1].min() - 1, pcs[:, 1].max() + 1)

# 3. Create 2 rows (top: > threshold, bottom: <= threshold) x N columns (one per threshold)
n_cols = len(THRESHOLDS)
fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 10), sharex=True, sharey=True)
axes = np.atleast_2d(axes)

for col_idx, thresh in enumerate(THRESHOLDS):
    above_mask = scores > thresh
    below_mask = ~above_mask

    # Top row: > threshold
    ax_top = axes[0, col_idx]
    ax_top.scatter(
        pcs[above_mask, 0], pcs[above_mask, 1],
        c="#d62728", s=8, alpha=0.65,
        label=f"n = {above_mask.sum()} ({above_mask.mean()*100:.1f}%)",
        rasterized=True
    )
    ax_top.set_title(f"Score > {thresh}", fontsize=11, fontweight="bold")
    ax_top.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax_top.set_xlim(xlim)
    ax_top.set_ylim(ylim)
    ax_top.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Bottom row: <= threshold
    ax_bottom = axes[1, col_idx]
    ax_bottom.scatter(
        pcs[below_mask, 0], pcs[below_mask, 1],
        c="#1f77b4", s=8, alpha=0.65,
        label=f"n = {below_mask.sum()} ({below_mask.mean()*100:.1f}%)",
        rasterized=True
    )
    ax_bottom.set_title(f"Score <= {thresh}", fontsize=11, fontweight="bold")
    ax_bottom.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax_bottom.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax_bottom.set_xlim(xlim)
    ax_bottom.set_ylim(ylim)
    ax_bottom.legend(loc="upper right", fontsize=9, framealpha=0.9)

fig.suptitle(f"{DATASET} Controls: Cytoplasmic_Ribo_Mito Score Splits", fontsize=13, fontweight="bold")
fig.tight_layout()

out_path = OUT_DIR / f"pca12_{DATASET}_controls_ribo_mito_thresholds.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Saved threshold split plot to: {out_path}")