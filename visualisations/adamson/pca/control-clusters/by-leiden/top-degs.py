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

# 1. Load data and subset to controls
adata_full = ad.read_h5ad(DATA_PATH)
adata_ctrl = adata_full[adata_full.obs["perturbation"] == CONTROL_LABEL].copy()

# 2. Run PCA & Leiden clustering
sc.pp.pca(adata_ctrl, n_comps=20)
pcs = adata_ctrl.obsm["X_pca"]
var_ratio = adata_ctrl.uns["pca"]["variance_ratio"]

sc.pp.neighbors(adata_ctrl, n_neighbors=30, n_pcs=10)
sc.tl.leiden(adata_ctrl, resolution=0.15, key_added="leiden_ctrl")

# 3. Differential expression between Leiden 0 and Leiden 1
sc.tl.rank_genes_groups(adata_ctrl, groupby="leiden_ctrl", method="wilcoxon", pts=True)
df_de = sc.get.rank_genes_groups_df(adata_ctrl, group="0")

# Rank by absolute Wilcoxon z-score (most statistically differentially expressed)
df_valid = df_de[
    (df_de["pvals_adj"] < 0.05) & np.isfinite(df_de["scores"])
].copy()
df_valid["abs_score"] = df_valid["scores"].abs()
top10 = df_valid.sort_values("abs_score", ascending=False).head(10).reset_index(drop=True)

print("=== TOP 10 MOST DIFFERENTIALLY EXPRESSED GENES (by |Wilcoxon score|) ===")
print(top10[["names", "scores", "logfoldchanges", "pvals_adj", "pct_nz_group", "pct_nz_reference"]].to_string(index=False))

# 4. Build single 3x4 figure:
#    [0, 0] Leiden clusters reference
#    [0, 1] Summary table of the top 10 most DE genes
#    [0, 2..3] & [1..2, 0..3] The 10 gene expression subplots
fig, axes = plt.subplots(3, 4, figsize=(22, 15))
axes_flat = axes.flatten()

# --- Subplot 1 (index 0): Leiden cluster reference ---
ax_ref = axes_flat[0]
for cl in sorted(adata_ctrl.obs["leiden_ctrl"].unique()):
    m = (adata_ctrl.obs["leiden_ctrl"] == cl).to_numpy()
    ax_ref.scatter(
        pcs[m, 0], pcs[m, 1],
        s=6, alpha=0.6,
        label=f"Leiden {cl} (n={m.sum()})",
        rasterized=True,
    )
ax_ref.set_title("Reference: Leiden Clusters", fontsize=11, fontweight="bold")
ax_ref.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
ax_ref.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
ax_ref.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

# --- Subplot 2 (index 1): Table of Top 10 most DE genes ---
ax_tbl = axes_flat[1]
ax_tbl.axis("off")
ax_tbl.set_title("Top 10 Most DE Genes (|Wilcoxon Score|)", fontsize=11, fontweight="bold", pad=8)

table_rows = []
for idx, row in top10.iterrows():
    direction = "Higher in 0" if row["scores"] > 0 else "Higher in 1"
    table_rows.append([
        f"{idx + 1}. {row['names']}",
        f"{row['scores']:+.1f}",
        f"{row['logfoldchanges']:+.2f}",
        direction,
    ])

tbl = ax_tbl.table(
    cellText=table_rows,
    colLabels=["Gene", "Score (z)", "log2FC", "Enriched"],
    loc="center",
    cellLoc="center",
    bbox=[0.0, 0.02, 1.0, 0.94],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2b3e50")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    elif r % 2 == 1:
        cell.set_facecolor("#f4f6f8")

# --- Subplots 3..12 (indices 2..11): Top 10 genes colored on PC1-PC2 ---
for i, row in top10.iterrows():
    ax = axes_flat[i + 2]
    gene = row["names"]
    score = row["scores"]
    lfc = row["logfoldchanges"]
    higher_in = "Leiden 0" if score > 0 else "Leiden 1"

    expr = adata_ctrl[:, gene].X
    if hasattr(expr, "toarray"):
        expr = expr.toarray().ravel()
    else:
        expr = np.asarray(expr).ravel()

    vmax = np.percentile(expr, 99)
    if vmax <= 0:
        vmax = expr.max() if expr.max() > 0 else 1.0

    sc_plot = ax.scatter(
        pcs[:, 0], pcs[:, 1],
        c=expr, cmap="viridis", vmin=0, vmax=vmax,
        s=6, alpha=0.7, rasterized=True,
    )
    cbar = plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7.5)

    ax.set_title(
        f"#{i + 1} {gene} | z={score:+.1f}, log2FC={lfc:+.2f} ({higher_in})",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=9)

fig.suptitle(
    f"{DATASET} Controls: Top 10 Most Differentially Expressed Genes (Leiden 0 vs Leiden 1)",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)
fig.tight_layout(rect=[0, 0, 1, 0.98])

out_path = OUT_DIR / f"pca12_{DATASET}_controls_top10_most_de.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"\nSaved combined figure to: {out_path}")