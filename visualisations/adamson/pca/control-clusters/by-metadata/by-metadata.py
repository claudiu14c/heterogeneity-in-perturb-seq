import math
import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

DATASET = "replogle22"
CONTROL_LABEL = "control"

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data and subset strictly to control cells
adata_full = ad.read_h5ad(DATA_PATH)
ctrl_mask = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
adata_ctrl = adata_full[ctrl_mask].copy()
print(f"Loaded {adata_ctrl.n_obs} control cells.")

# 2. Run PCA on controls only
sc.pp.pca(adata_ctrl, n_comps=20)
pcs = adata_ctrl.obsm["X_pca"]
var_ratio = adata_ctrl.uns["pca"]["variance_ratio"]

# 3. Print top genes driving PC1 and PC2
loadings = adata_ctrl.varm["PCs"]
for pc_idx in [0, 1]:
    top_idx = np.argsort(np.abs(loadings[:, pc_idx]))[::-1][:10]
    top_genes = adata_ctrl.var_names[top_idx].tolist()
    print(f"Top 10 genes driving PC{pc_idx + 1}: {top_genes}")

# 4. Identify plottable columns in adata_ctrl.obs using pandas dtype checks
plottable_cols = []
for col in adata_ctrl.obs.columns:
    if col == "perturbation":
        continue
    series = adata_ctrl.obs[col]
    n_unique = series.nunique(dropna=False)
    is_numeric = pd.api.types.is_numeric_dtype(series)

    # Include numeric QC metrics OR categorical columns with 2 to 30 unique values
    if (is_numeric and n_unique > 1) or (not is_numeric and 1 < n_unique <= 30):
        plottable_cols.append(col)
    else:
        print(f"Skipping '{col}' (unique values: {n_unique}, dtype: {series.dtype})")

print(f"\nPlotting PCA colored by columns: {plottable_cols}")

# 5. Build dynamic grid of subplots
n_plots = max(len(plottable_cols), 1)
n_cols = 3
n_rows = math.ceil(n_plots / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = np.atleast_1d(axes).flatten()

for idx, col in enumerate(plottable_cols):
    ax = axes[idx]
    series = adata_ctrl.obs[col]

    # Plot continuous numeric metadata (e.g. UMI counts, gene counts, mito %)
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 15:
        sc_plot = ax.scatter(
            pcs[:, 0], pcs[:, 1],
            c=series.to_numpy(), cmap="viridis",
            s=8, alpha=0.6, rasterized=True
        )
        plt.colorbar(sc_plot, ax=ax, label=col)
    # Plot discrete categories (e.g. gem_group, batch, guide_id, condition)
    else:
        categories = series.astype(str).unique()
        cmap = plt.get_cmap("tab20" if len(categories) > 10 else "tab10")
        for c_idx, cat in enumerate(sorted(categories)):
            mask = (series.astype(str) == cat).to_numpy()
            ax.scatter(
                pcs[mask, 0], pcs[mask, 1],
                c=[cmap(c_idx % cmap.N)],
                s=8, alpha=0.6,
                label=f"{cat} (n={mask.sum()})",
                rasterized=True
            )
        ax.legend(loc="best", fontsize=8, framealpha=0.85, markerscale=1.5)

    ax.set_title(f"Colored by: {col}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")

# Hide any unused subplot frames
for empty_idx in range(len(plottable_cols), len(axes)):
    axes[empty_idx].axis("off")

fig.suptitle(f"Dataset: {DATASET} | Control Cells PCA Diagnostics (n={adata_ctrl.n_obs})", fontsize=14)
fig.tight_layout()

out_path = OUT_DIR / f"pca12_{DATASET}_controls_metadata_diagnostics.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Saved diagnostic plot to: {out_path}")