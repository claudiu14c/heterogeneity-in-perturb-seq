
import os
from pathlib import Path

import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DATASET = "adamson16"
CONTROL_LABEL = "control"

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/"
    f"{DATASET}/{DATASET}_processed.h5ad",
))

OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Load data and subset controls
# ============================================================

print("Loading dataset...")

adata_full = ad.read_h5ad(DATA_PATH)

adata = adata_full[
    adata_full.obs["perturbation"] == CONTROL_LABEL
].copy()

print(f"Control cells: {adata.n_obs}")
print(f"Genes: {adata.n_vars}")

print("First gene names:", adata.var_names[:10].tolist())


# ============================================================
# 2. Calculate PCA
# ============================================================

print("\nCalculating PCA...")

sc.pp.pca(adata, n_comps=20)

pcs = adata.obsm["X_pca"]
var_ratio = adata.uns["pca"]["variance_ratio"]


# ============================================================
# 3. Define G0 / quiescence-associated markers
# ============================================================

# Human gene symbols.
# These are candidate quiescence-associated markers,
# not definitive G0-specific markers.

g0_genes = [
    "CDKN1B",
    "CDKN1A",
    "BTG1",
    "BTG2",
    "KLF4",
    "FOXO3",
    "GADD45A",
    "GADD45B",
    "TXNIP",
    "SESN1",
    "SESN2",
    "DUSP1",
    "ZFP36",
    "IER2",
    "IER3",
]


# Proliferation-associated markers

proliferation_genes = [
    "MKI67",
    "PCNA",
    "MCM2",
    "MCM5",
    "MCM6",
    "CDK1",
    "CCNB1",
    "CCNB2",
    "TOP2A",
    "UBE2C",
    "CDC20",
    "AURKA",
    "AURKB",
    "TYMS",
    "RRM2",
]


# ============================================================
# 4. Match genes to dataset
# ============================================================

gene_names = set(adata.var_names)

g0_genes = [
    g for g in g0_genes
    if g in gene_names
]

proliferation_genes = [
    g for g in proliferation_genes
    if g in gene_names
]

print("\nG0-associated genes found:")
print(len(g0_genes), g0_genes)

print("\nProliferation genes found:")
print(len(proliferation_genes), proliferation_genes)

if len(g0_genes) < 3 or len(proliferation_genes) < 3:
    raise ValueError(
        "Too few marker genes found. Check gene identifiers "
        "and species."
    )


# ============================================================
# 5. Calculate gene scores
# ============================================================

# Assumes adata.X contains normalized, log-transformed
# expression values.

print("\nCalculating G0 and proliferation scores...")

sc.tl.score_genes(
    adata,
    gene_list=g0_genes,
    score_name="G0_score",
    use_raw=False,
)

sc.tl.score_genes(
    adata,
    gene_list=proliferation_genes,
    score_name="proliferation_score",
    use_raw=False,
)

print("\nScore summary:")
print(
    adata.obs[
        ["G0_score", "proliferation_score"]
    ].describe()
)


# ============================================================
# 6. Identify candidate G0 cells
# ============================================================

# Candidate G0:
#   G0 score above the median
#   AND proliferation score below the median

g0_threshold = adata.obs["G0_score"].median()

prolif_threshold = (
    adata.obs["proliferation_score"].median()
)

adata.obs["G0_candidate"] = (
    (adata.obs["G0_score"] > g0_threshold)
    &
    (adata.obs["proliferation_score"] < prolif_threshold)
)

print("\nCandidate G0 classification:")
print(adata.obs["G0_candidate"].value_counts())

g0_percentage = (
    adata.obs["G0_candidate"].mean() * 100
)

print(
    f"\nCandidate G0 proportion: {g0_percentage:.2f}%"
)


# ============================================================
# 7. Plot PCA colored by G0 state
# ============================================================

print("\nGenerating G0 PCA plot...")

fig, ax = plt.subplots(figsize=(7, 6))

colors = {
    False: "lightgrey",
    True: "darkorange",
}

for state in [False, True]:

    mask = (
        adata.obs["G0_candidate"] == state
    ).to_numpy()

    ax.scatter(
        pcs[mask, 0],
        pcs[mask, 1],
        s=8,
        alpha=0.65,
        color=colors[state],
        label=(
            f"G0 candidate (n={mask.sum()})"
            if state
            else f"Non-G0 (n={mask.sum()})"
        ),
        rasterized=True,
    )

ax.set_xlabel(
    f"PC1 ({var_ratio[0] * 100:.1f}%)"
)

ax.set_ylabel(
    f"PC2 ({var_ratio[1] * 100:.1f}%)"
)

ax.set_title(
    f"{DATASET} Controls: G0 State",
    fontweight="bold",
)

ax.legend(loc="best")

fig.tight_layout()

out_path = (
    OUT_DIR / f"pca12_{DATASET}_controls_G0.png"
)

fig.savefig(
    out_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Saved G0 PCA plot: {out_path}")


# ============================================================
# 8. Save annotated AnnData
# ============================================================

results_path = (
    OUT_DIR / f"{DATASET}_controls_G0_scoring.h5ad"
)

adata.write(results_path)

print(f"Saved annotated AnnData: {results_path}")


# ============================================================
# Finished
# ============================================================

print("\n========================================")
print("G0 analysis completed!")
print("========================================")
print(f"Total control cells: {adata.n_obs}")
print(f"Candidate G0 cells: {adata.obs['G0_candidate'].sum()}")
print(f"Candidate G0 proportion: {g0_percentage:.2f}%")
print(f"Output directory: {OUT_DIR}")

# First gene names: ['FAM87B', 'LINC00115', 'RP11-54O7.16', 'SAMD11', 'HES4', 'ISG15', 'AGRN', 'C1orf159', 'TTLL10', 'TNFRSF4']

# Calculating PCA...

# G0-associated genes found:
# 12 ['CDKN1B', 'CDKN1A', 'BTG1', 'BTG2', 'GADD45A', 'GADD45B', 'TXNIP', 'SESN2', 'DUSP1', 'ZFP36', 'IER2', 'IER3']

# Proliferation genes found:
# 14 ['MKI67', 'PCNA', 'MCM2', 'MCM5', 'MCM6', 'CDK1', 'CCNB1', 'CCNB2', 'TOP2A', 'CDC20', 'AURKA', 'AURKB', 'TYMS', 'RRM2']

# Calculating G0 and proliferation scores...

# Score summary:
#           G0_score  proliferation_score
# count  7216.000000          7216.000000
# mean      0.068890             0.083151
# std       0.136979             0.208810
# min      -0.315969            -0.436179
# 25%      -0.015388            -0.067543
# 50%       0.052275             0.095609
# 75%       0.131458             0.226150
# max       1.292236             0.927848

# Candidate G0 classification:
# G0_candidate
# False    5363
# True     1853
# Name: count, dtype: int64

# Candidate G0 proportion: 25.68%

# Generating G0 PCA plot...
# Saved G0 PCA plot: pca12_adamson16_controls_G0.png
# Saved annotated AnnData: adamson16_controls_G0_scoring.h5ad

# ========================================
# G0 analysis completed!
# ========================================
# Total control cells: 7216
# Candidate G0 cells: 1853
# Candidate G0 proportion: 