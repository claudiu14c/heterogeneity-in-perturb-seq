import os
import re
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
CELLTYPE_COL = "cell_line"
TARGET_CLUSTERS = 2

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"../../../../data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_leiden_for_k_clusters(
    adata: ad.AnnData,
    target_k: int = 2,
    key_added: str = "leiden_ctrl",
    res_min: float = 0.001,
    res_max: float = 0.5,
    max_iter: int = 25,
) -> float:
    """Binary-search Leiden resolution to obtain exactly `target_k` clusters."""
    best_res = res_min
    best_diff = float("inf")
    best_labels = None

    low, high = res_min, res_max
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        sc.tl.leiden(
            adata,
            resolution=mid,
            key_added=key_added,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        k = adata.obs[key_added].nunique()
        diff = abs(k - target_k)

        if diff < best_diff:
            best_diff = diff
            best_res = mid
            best_labels = adata.obs[key_added].copy()

        if k == target_k:
            return mid
        elif k < target_k:
            low = mid
        else:
            high = mid

    if best_labels is not None:
        adata.obs[key_added] = best_labels
    return best_res


def run_leiden_de_for_cell_line(adata_cl: ad.AnnData, cell_line_name: str) -> None:
    safe_cl_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(cell_line_name)).strip("_")
    print(f"\n{'='*60}\nProcessing cell line: {cell_line_name} (total cells={adata_cl.n_obs})\n{'='*60}")

    # 1. Compute HVGs on ALL cells of this cell line
    print(f"[{cell_line_name}] Computing HVGs across all {adata_cl.n_obs} cells...")
    sc.pp.highly_variable_genes(adata_cl, n_top_genes=2000, flavor="seurat")
    n_hvgs = int(adata_cl.var["highly_variable"].sum())
    print(f"[{cell_line_name}] Number of HVGs selected: {n_hvgs}")

    # 2. Subset to controls (retaining cell-line-wide HVG mask and full gene set for DE)
    adata_ctrl = adata_cl[adata_cl.obs["perturbation"] == CONTROL_LABEL].copy()
    print(f"[{cell_line_name}] Control cells: {adata_ctrl.n_obs}")

    if adata_ctrl.n_obs < 10:
        print(f"[{cell_line_name}] WARNING: Too few control cells to run PCA/Leiden. Skipping.")
        return

    # 3. Run PCA on controls using cell-line-wide HVGs, then Leiden tuned to 2 clusters
    n_comps = min(20, adata_ctrl.n_obs - 1, n_hvgs - 1)
    sc.pp.pca(adata_ctrl, n_comps=n_comps, mask_var="highly_variable")

    n_pcs_use = min(12, n_comps)
    sc.pp.neighbors(adata_ctrl, n_neighbors=30, n_pcs=n_pcs_use)

    chosen_res = find_leiden_for_k_clusters(
        adata_ctrl,
        target_k=TARGET_CLUSTERS,
        key_added="leiden_ctrl",
    )
    clusters = sorted(adata_ctrl.obs["leiden_ctrl"].unique())
    print(f"[{cell_line_name}] Leiden clusters found: {clusters} (resolution={chosen_res:.5f})")

    if len(clusters) != 2:
        print(f"[{cell_line_name}] WARNING: Could not isolate exactly 2 clusters (got {clusters}).")
        if len(clusters) < 2:
            return

    # 4. Run t-SNE on the first 12 PCs for visualization
    print(f"[{cell_line_name}] Running t-SNE on the first {n_pcs_use} PCs...")
    sc.tl.tsne(
        adata_ctrl,
        use_rep="X_pca",
        n_pcs=n_pcs_use,
        random_state=42,
    )
    tsne_coords = adata_ctrl.obsm["X_tsne"]

    # 5. Differential expression between Leiden 0 and Leiden 1 across all genes
    sc.tl.rank_genes_groups(
        adata_ctrl,
        groupby="leiden_ctrl",
        groups=["0"],
        reference="1",
        method="wilcoxon",
        pts=True,
    )
    df_de = sc.get.rank_genes_groups_df(adata_ctrl, group="0")

    if "pct_nz_reference" not in df_de.columns and "1" in adata_ctrl.uns["rank_genes_groups"]["pts"]:
        pts_df = adata_ctrl.uns["rank_genes_groups"]["pts"]
        df_de["pct_nz_reference"] = df_de["names"].map(pts_df["1"])

    df_valid = df_de[(df_de["pvals_adj"] < 0.05) & np.isfinite(df_de["scores"])].copy()

    if df_valid.empty:
        print(f"[{cell_line_name}] WARNING: No significant DE genes (pvals_adj < 0.05) found. Skipping plot.")
        return

    # Take Top 5 most enriched in Leiden 0 (highest positive score)
    # + Top 5 most enriched in Leiden 1 (most negative score)
    top5_c0 = df_valid[df_valid["scores"] > 0].sort_values("scores", ascending=False).head(5)
    top5_c1 = df_valid[df_valid["scores"] < 0].sort_values("scores", ascending=True).head(5)
    top10 = pd.concat([top5_c0, top5_c1], ignore_index=True)

    print(f"\n=== [{cell_line_name}] TOP 5 LEIDEN 0 + TOP 5 LEIDEN 1 MOST DE GENES ===")
    display_cols = [
        c for c in ["names", "scores", "logfoldchanges", "pvals_adj", "pct_nz_group", "pct_nz_reference"]
        if c in top10.columns
    ]
    print(top10[display_cols].to_string(index=False))

    # 6. Build single 3x4 t-SNE figure:
    #    [0, 0] Leiden clusters reference (t-SNE)
    #    [0, 1] Summary table (Top 5 Leiden 0 + Top 5 Leiden 1)
    #    [0, 2..3] & [1..2, 0..3] The up to 10 gene expression t-SNE subplots
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    axes_flat = axes.flatten()

    # --- Subplot 1 (index 0): Leiden cluster reference in t-SNE space ---
    ax_ref = axes_flat[0]
    for cl in clusters:
        m = (adata_ctrl.obs["leiden_ctrl"] == cl).to_numpy()
        ax_ref.scatter(
            tsne_coords[m, 0], tsne_coords[m, 1],
            s=6, alpha=0.6,
            label=f"Leiden {cl} (n={m.sum()})",
            rasterized=True,
        )
    ax_ref.set_title(
        f"Reference: Leiden Clusters ({cell_line_name}, res={chosen_res:.4f})",
        fontsize=11,
        fontweight="bold",
    )
    ax_ref.set_xlabel("t-SNE 1")
    ax_ref.set_ylabel("t-SNE 2")
    ax_ref.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # --- Subplot 2 (index 1): Table of Top 5 Leiden 0 + Top 5 Leiden 1 ---
    ax_tbl = axes_flat[1]
    ax_tbl.axis("off")
    ax_tbl.set_title("Top 5 Leiden 0 & Top 5 Leiden 1 DE Genes", fontsize=11, fontweight="bold", pad=8)

    table_rows = []
    for idx, row in top10.iterrows():
        direction = "Leiden 0" if row["scores"] > 0 else "Leiden 1"
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
        elif r <= len(top5_c0):
            cell.set_facecolor("#eef5fc")
        else:
            cell.set_facecolor("#fdf2e9")

    # --- Subplots 3..12 (indices 2..11): Leiden 0 & Leiden 1 markers in t-SNE space ---
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
            tsne_coords[:, 0], tsne_coords[:, 1],
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
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)

    for j in range(len(top10) + 2, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"{DATASET} ({cell_line_name}) Controls t-SNE: Top 5 DE Genes per Leiden Cluster (Leiden 0 vs Leiden 1)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = OUT_DIR / f"tsne12_{DATASET}_{safe_cl_name}_controls_top5_each_leiden.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[{cell_line_name}] Saved combined t-SNE figure to: {out_path}")


# --- Main Execution ---
print(f"Loading dataset from: {DATA_PATH}")
adata_full = ad.read_h5ad(DATA_PATH)

if CELLTYPE_COL not in adata_full.obs.columns:
    raise KeyError(
        f"Column '{CELLTYPE_COL}' not found in adata.obs. Available columns: {list(adata_full.obs.columns)}"
    )

cell_lines = adata_full.obs[CELLTYPE_COL].dropna().unique().tolist()
print(f"Found {len(cell_lines)} cell lines in '{CELLTYPE_COL}': {cell_lines}")

for cl in cell_lines:
    adata_cl_copy = adata_full[adata_full.obs[CELLTYPE_COL] == cl].copy()
    run_leiden_de_for_cell_line(adata_cl_copy, cl)

print("\nDone!")