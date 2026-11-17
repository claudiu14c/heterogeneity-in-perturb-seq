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

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

adata_full = ad.read_h5ad(DATA_PATH)
adata_ctrl = adata_full[adata_full.obs["perturbation"] == CONTROL_LABEL].copy()

# 1. Define canonical stress & pathway gene sets
pathway_genes = {
    "UPR_ER_Stress (Tm/Tg)": [
        "HSPA5", "DDIT3", "HERPUD1", "XBP1", "ATF4", "DNAJB9",
        "HYOU1", "SDF2L1", "PDIA4", "CALR", "HSP90B1", "MANF"
    ],
    "ISR_Stress": [
        "ATF4", "DDIT3", "PPP1R15A", "ASNS", "TRIB3", "SLC7A11", "GDF15"
    ],
    "Cell_Cycle_S_G2M": [
        "MKI67", "TOP2A", "CDK1", "PCNA", "HIST1H1E", "HIST1H1D",
        "UBE2C", "BIRC5", "TYMS", "RRM2"
    ],
    "Cytoplasmic_Ribo_Mito": [
        "MT-CO3", "MT-CYB", "MT-CO2", "MT-ND4", "RPL21", "RPS23",
        "RPL12", "RPS3A", "RPL7", "RPS6"
    ],
}

# 2. Score each pathway (filtering to genes present in adata_ctrl.var_names)
score_cols = []
for pathway_name, genes in pathway_genes.items():
    present = [g for g in genes if g in adata_ctrl.var_names]
    print(f"{pathway_name}: {len(present)}/{len(genes)} genes found -> {present}")
    if len(present) >= 2:
        col_name = f"score_{pathway_name}"
        sc.tl.score_genes(adata_ctrl, gene_list=present, score_name=col_name)
        score_cols.append(col_name)

# 3. Run PCA on controls
sc.pp.pca(adata_ctrl, n_comps=20)
pcs = adata_ctrl.obsm["X_pca"]
var_ratio = adata_ctrl.uns["pca"]["variance_ratio"]

# 4. Plot PCA colored by each stress/pathway score
fig, axes = plt.subplots(1, len(score_cols), figsize=(6 * len(score_cols), 5))
axes = np.atleast_1d(axes).flatten()

for idx, col in enumerate(score_cols):
    ax = axes[idx]
    vals = adata_ctrl.obs[col].to_numpy()
    # Clip extreme 1st/99th percentiles so outliers don't wash out the color gradient
    vmin, vmax = np.percentile(vals, [1, 99])
    sc_plot = ax.scatter(
        pcs[:, 0], pcs[:, 1],
        c=vals, cmap="coolwarm", vmin=vmin, vmax=vmax,
        s=8, alpha=0.65, rasterized=True
    )
    plt.colorbar(sc_plot, ax=ax, label="Module Score")
    ax.set_title(col.replace("score_", ""), fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")

fig.suptitle(f"{DATASET} Controls: Stress & Pathway Module Scores", fontsize=13)
fig.tight_layout()

out_path = OUT_DIR / f"pca12_{DATASET}_controls_stress_pathways.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Saved pathway score plot to: {out_path}")