
import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATASET = "adamson16"
CONTROL_LABEL = "control"

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))

OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Load data and subset controls
# ============================================================

print("Loading dataset...")

adata_full = ad.read_h5ad(DATA_PATH)

adata_ctrl = adata_full[
    adata_full.obs["perturbation"] == CONTROL_LABEL
].copy()

print(f"Control cells: {adata_ctrl.n_obs}")
print(f"Genes: {adata_ctrl.n_vars}")

print("First gene names:", adata_ctrl.var_names[:10].tolist())


# ============================================================
# 2. PCA and Leiden clustering
# ============================================================

print("\nRunning PCA and Leiden clustering...")

sc.pp.pca(adata_ctrl, n_comps=20)

pcs = adata_ctrl.obsm["X_pca"]
var_ratio = adata_ctrl.uns["pca"]["variance_ratio"]

sc.pp.neighbors(
    adata_ctrl,
    n_neighbors=30,
    n_pcs=10,
)

sc.tl.leiden(
    adata_ctrl,
    resolution=0.15,
    key_added="leiden_ctrl",
)

print("\nLeiden cluster counts:")
print(adata_ctrl.obs["leiden_ctrl"].value_counts())


# ============================================================
# 3. Differential expression between Leiden clusters
# ============================================================

print("\nRunning differential expression...")

sc.tl.rank_genes_groups(
    adata_ctrl,
    groupby="leiden_ctrl",
    method="wilcoxon",
)

df_c0 = sc.get.rank_genes_groups_df(
    adata_ctrl,
    group="0",
)

print("\n=== TOP 15 GENES HIGHER IN LEIDEN 0 ===")

print(
    df_c0.head(15)[
        ["names", "scores", "logfoldchanges", "pvals_adj"]
    ].to_string(index=False)
)

print("\n=== TOP 15 GENES HIGHER IN LEIDEN 1 ===")

print(
    df_c0.tail(15).iloc[::-1][
        ["names", "scores", "logfoldchanges", "pvals_adj"]
    ].to_string(index=False)
)


# ============================================================
# 4. Plot PCA colored by Leiden clusters
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

for cl in sorted(adata_ctrl.obs["leiden_ctrl"].unique()):
    mask = (
        adata_ctrl.obs["leiden_ctrl"] == cl
    ).to_numpy()

    ax.scatter(
        pcs[mask, 0],
        pcs[mask, 1],
        s=8,
        alpha=0.6,
        label=f"Leiden {cl} (n={mask.sum()})",
        rasterized=True,
    )

ax.set_title(
    f"{DATASET} Controls: Leiden Clustering (resolution=0.15)",
    fontweight="bold",
)
ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
ax.legend(loc="lower right")

fig.tight_layout()

out_path = OUT_DIR / f"pca12_{DATASET}_controls_leiden.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)

print(f"\nSaved Leiden PCA: {out_path}")


# ============================================================
# 5. Offline cell-cycle marker gene lists
# ============================================================

# Human cell-cycle marker genes.
# No external downloads required.

s_genes = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4",
    "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL",
    "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP",
    "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2",
    "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN",
    "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN",
    "POLA1", "CHAF1B", "BRIP1", "E2F8"
]

g2m_genes = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5",
    "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2",
    "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3",
    "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2",
    "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B",
    "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1",
    "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2",
    "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN",
    "LBR", "CKAP5", "CENPE", "CTCF", "NEK2",
    "G2E3", "GAS2L3", "CBX5", "CENPA"
]

# Match genes to dataset identifiers.
# The lists above use uppercase human gene symbols.

gene_names = set(adata_ctrl.var_names)

s_genes = [
    gene for gene in s_genes
    if gene in gene_names
]

g2m_genes = [
    gene for gene in g2m_genes
    if gene in gene_names
]

print("\nCell-cycle marker genes found:")
print(f"S phase: {len(s_genes)}")
print(f"G2M phase: {len(g2m_genes)}")

if len(s_genes) < 5 or len(g2m_genes) < 5:
    raise ValueError(
        "Too few cell-cycle genes found. "
        "Check whether the dataset uses mouse or human gene symbols "
        "and whether the expression matrix contains the required genes."
    )


# ============================================================
# 6. Cell-cycle scoring
# ============================================================

print("\nScoring cell-cycle phases...")

sc.tl.score_genes_cell_cycle(
    adata_ctrl,
    s_genes=s_genes,
    g2m_genes=g2m_genes,
)

print("\nCell-cycle phase counts:")
print(adata_ctrl.obs["phase"].value_counts())


# ============================================================
# 7. Plot PCA colored by cell-cycle phase
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

phase_colors = {
    "G1": "steelblue",
    "S": "orange",
    "G2M": "crimson",
}

for phase in ["G1", "S", "G2M"]:

    mask = (
        adata_ctrl.obs["phase"] == phase
    ).to_numpy()

    ax.scatter(
        pcs[mask, 0],
        pcs[mask, 1],
        s=8,
        alpha=0.6,
        color=phase_colors[phase],
        label=f"{phase} (n={mask.sum()})",
        rasterized=True,
    )

ax.set_title(
    f"{DATASET} Controls: Cell-cycle Phases",
    fontweight="bold",
)
ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
ax.legend(loc="best")

fig.tight_layout()

out_path = OUT_DIR / f"pca12_{DATASET}_controls_cellcycle.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)

print(f"Saved cell-cycle PCA: {out_path}")


# ============================================================
# 8. Plot continuous S and G2M scores
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, score in zip(
    axes,
    ["S_score", "G2M_score"],
):

    sc.pl.pca(
        adata_ctrl,
        color=score,
        ax=ax,
        show=False,
        size=8,
        alpha=0.7,
        color_map="viridis",
        title=score,
    )

fig.tight_layout()

out_path = OUT_DIR / f"pca12_{DATASET}_controls_cellcycle_scores.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)

print(f"Saved cell-cycle score PCA: {out_path}")


# ============================================================
# 9. Cell-cycle composition by Leiden cluster
# ============================================================

phase_composition = pd.crosstab(
    adata_ctrl.obs["leiden_ctrl"],
    adata_ctrl.obs["phase"],
    normalize="index",
) * 100

print("\nCell-cycle composition by Leiden cluster (%):")
print(phase_composition.round(2))

# Stacked bar plot

ax = phase_composition.plot(
    kind="bar",
    stacked=True,
    figsize=(7, 5),
    color=[
        phase_colors.get(c, "grey")
        for c in phase_composition.columns
    ],
)

ax.set_ylabel("Cells (%)")
ax.set_xlabel("Leiden cluster")
ax.set_title("Cell-cycle composition by Leiden cluster")
ax.legend(
    title="Cell-cycle phase",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

plt.tight_layout()

out_path = OUT_DIR / f"{DATASET}_cellcycle_by_cluster.png"
plt.savefig(out_path, dpi=200)
plt.close()

print(f"Saved cluster composition: {out_path}")


# ============================================================
# 10. Compare continuous cell-cycle scores by cluster
# ============================================================

score_df = adata_ctrl.obs[
    ["leiden_ctrl", "S_score", "G2M_score"]
].copy()

print("\nMean cell-cycle scores by Leiden cluster:")
print(
    score_df.groupby("leiden_ctrl")[
        ["S_score", "G2M_score"]
    ].agg(["mean", "median", "std"]).round(4)
)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, score in zip(
    axes,
    ["S_score", "G2M_score"],
):

    clusters = sorted(score_df["leiden_ctrl"].unique())

    values = [
        score_df.loc[
            score_df["leiden_ctrl"] == cl,
            score,
        ].dropna().values
        for cl in clusters
    ]

    ax.boxplot(
        values,
        tick_labels=clusters,
        showfliers=False,
    )

    ax.set_xlabel("Leiden cluster")
    ax.set_ylabel(score)
    ax.set_title(f"{score} by Leiden cluster")

fig.tight_layout()

out_path = OUT_DIR / f"{DATASET}_cellcycle_scores_by_cluster.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)

print(f"Saved score comparison: {out_path}")


# ============================================================
# 11. Save results
# ============================================================

results_path = OUT_DIR / f"{DATASET}_controls_cellcycle.h5ad"

adata_ctrl.write(results_path)

print("\n========================================")
print("Analysis completed successfully!")
print("========================================")
print(f"Results directory: {OUT_DIR}")
print(f"Annotated AnnData: {results_path}")
