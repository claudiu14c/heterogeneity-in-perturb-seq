
import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

print("Start")

# ============================================================
# 0. SETTINGS
# ============================================================

DATASET = "adamson16"
CONTROL_LABEL = "control"

N_HVGS = 2000
N_PCS = 12
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 0.3

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))

OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")

adata_full = ad.read_h5ad(DATA_PATH)

print(f"Full dataset shape: {adata_full.shape}")

# ============================================================
# 1. SELECT CONTROL CELLS
# ============================================================

ctrl_mask = (
    adata_full.obs["perturbation"] == CONTROL_LABEL
).values

adata_ctrl = adata_full[ctrl_mask].copy()

print(f"Number of control cells: {adata_ctrl.n_obs}")

# ============================================================
# 2. COMPUTE HVGs GLOBALLY
#    Keep the FULL gene matrix for differential expression.
# ============================================================

print("Computing global HVGs...")

sc.pp.highly_variable_genes(
    adata_full,
    n_top_genes=N_HVGS,
    flavor="seurat"
)

hvg_mask = adata_full.var["highly_variable"].values

print(f"Selected HVGs: {hvg_mask.sum()}")

# Subset control cells to HVGs ONLY for PCA / clustering.
adata_ctrl_hvg = adata_ctrl[:, hvg_mask].copy()

# ============================================================
# 3. PCA ON CONTROL CELLS
# ============================================================

print("Running PCA...")

sc.pp.pca(
    adata_ctrl_hvg,
    n_comps=50,
    random_state=42
)

pcs_ctrl = adata_ctrl_hvg.obsm["X_pca"]

var_ratio_ctrl = adata_ctrl_hvg.uns["pca"]["variance_ratio"]

# ============================================================
# 4. t-SNE ON FIRST 12 PCs
# ============================================================

print("Running t-SNE...")

sc.tl.tsne(
    adata_ctrl_hvg,
    n_pcs=N_PCS,
    random_state=42
)

tsne_coords = adata_ctrl_hvg.obsm["X_tsne"]

# ============================================================
# 5. BUILD NEIGHBOR GRAPH AND RUN LEIDEN
# ============================================================

print("Constructing PCA-based neighborhood graph...")

sc.pp.neighbors(
    adata_ctrl_hvg,
    n_neighbors=N_NEIGHBORS,
    n_pcs=N_PCS,
    random_state=42
)

print("Running Leiden clustering...")

sc.tl.leiden(
    adata_ctrl_hvg,
    resolution=LEIDEN_RESOLUTION,
    key_added="cluster",
    random_state=42
)

# Transfer cluster labels to the full-gene control object.
adata_ctrl.obs["cluster"] = (
    adata_ctrl_hvg.obs["cluster"].astype(str).values
)

print("\nCluster composition:")
print(adata_ctrl.obs["cluster"].value_counts())

n_clusters = adata_ctrl.obs["cluster"].nunique()

print(f"Number of Leiden clusters: {n_clusters}")

# ============================================================
# 6. PLOT t-SNE COLORED BY CLUSTER
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

cluster_labels = adata_ctrl_hvg.obs["cluster"].astype(str)

for cluster in sorted(cluster_labels.unique()):
    mask = cluster_labels.values == cluster

    ax.scatter(
        tsne_coords[mask, 0],
        tsne_coords[mask, 1],
        s=6,
        alpha=0.6,
        label=f"Cluster {cluster} (n={mask.sum()})",
        rasterized=True
    )

ax.set_title(
    f"Control cells: t-SNE ({N_PCS} PCs) | {DATASET}"
)

ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")

ax.legend(
    loc="upper right",
    framealpha=0.9,
    fontsize=9
)

# PCA inset
ax_inset = ax.inset_axes([0.03, 0.62, 0.32, 0.32])

for cluster in sorted(cluster_labels.unique()):
    mask = cluster_labels.values == cluster

    ax_inset.scatter(
        pcs_ctrl[mask, 0],
        pcs_ctrl[mask, 1],
        s=2,
        alpha=0.5,
        label=cluster
    )

ax_inset.set_title("PC1 vs PC2", fontsize=8)
ax_inset.set_xlabel(
    f"PC1 ({var_ratio_ctrl[0]*100:.1f}%)",
    fontsize=7
)
ax_inset.set_ylabel(
    f"PC2 ({var_ratio_ctrl[1]*100:.1f}%)",
    fontsize=7
)

ax_inset.tick_params(labelsize=6)

fig.tight_layout()

tsne_out = OUT_DIR / "tsne_control_clusters.png"

fig.savefig(tsne_out, dpi=200)
plt.close(fig)

print("Saved:", tsne_out)

# ============================================================
# 7. DIFFERENTIAL EXPRESSION BETWEEN CLUSTERS
#    Test ALL genes, not only HVGs.
# ============================================================

if n_clusters != 2:
    print(
        "\nWARNING: Leiden did not identify exactly two clusters."
        "\nAdjust LEIDEN_RESOLUTION and inspect the t-SNE plot."
        "\nSkipping the two-cluster DE comparison."
    )

else:

    cluster_ids = sorted(
        adata_ctrl.obs["cluster"].unique()
    )

    cluster_1, cluster_2 = cluster_ids

    print(
        f"\nDifferential expression: "
        f"Cluster {cluster_1} vs Cluster {cluster_2}"
    )

    # Assumes adata_ctrl.X contains normalized log-expression.
    # use_raw=False ensures that the full gene matrix is tested.

    sc.tl.rank_genes_groups(
        adata_ctrl,
        groupby="cluster",
        groups=[cluster_1],
        reference=cluster_2,
        method="wilcoxon",
        use_raw=False
    )

    de = sc.get.rank_genes_groups_df(
        adata_ctrl,
        group=cluster_1
    )

    de = de.sort_values("pvals_adj")

    de.to_csv(
        OUT_DIR / "DE_cluster1_vs_cluster2.csv",
        index=False
    )

    print("\nTop 30 differentially expressed genes:")
    print(de.head(30).to_string(index=False))

    # ========================================================
    # 8. VOLCANO PLOT
    # ========================================================

    de["neg_log10_padj"] = -np.log10(
        de["pvals_adj"].clip(lower=1e-300)
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        de["logfoldchanges"],
        de["neg_log10_padj"],
        s=8,
        alpha=0.5
    )

    ax.axhline(
        -np.log10(0.05),
        linestyle="--"
    )

    ax.axvline(0, linestyle="--")

    ax.set_xlabel("Log2 fold change")
    ax.set_ylabel("-Log10 adjusted p-value")

    ax.set_title(
        f"Cluster {cluster_1} vs Cluster {cluster_2}"
    )

    fig.tight_layout()

    volcano_out = OUT_DIR / "volcano_clusters.png"

    fig.savefig(volcano_out, dpi=200)
    plt.close(fig)

    print("Saved:", volcano_out)

    # ========================================================
    # 9. HEATMAP OF TOP DIFFERENTIALLY EXPRESSED GENES
    # ========================================================

    top_genes = (
        de.loc[de["pvals_adj"] < 0.05]
        .head(30)["names"]
        .tolist()
    )

    if len(top_genes) > 0:

        sc.pl.heatmap(
            adata_ctrl,
            var_names=top_genes,
            groupby="cluster",
            use_raw=False,
            swap_axes=True,
            show=False
        )

        plt.savefig(
            OUT_DIR / "heatmap_top_DE_genes.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close("all")

        print("Saved heatmap.")

    else:
        print("No genes passed the adjusted p-value threshold.")

    # ========================================================
    # 10. SAVE CELL CLUSTER ASSIGNMENTS
    # ========================================================

    adata_ctrl.write_h5ad(
        OUT_DIR / "control_cells_with_clusters.h5ad"
    )

    print("\nDifferential expression analysis completed.")

print("\nDone!")
