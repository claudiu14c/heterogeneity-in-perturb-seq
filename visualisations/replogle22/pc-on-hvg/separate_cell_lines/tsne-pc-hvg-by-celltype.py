import os
import re
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

print("Start")

DATASET = "replogle22"
PERTURBATIONS = ['PAFAH1B1', 'SEM1', 'RPA3', 'NOP9']
CONTROL_LABEL = "control"
CELLTYPE_COL = "cell_line"

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"../../../data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
adata_raw = ad.read_h5ad(DATA_PATH)
print(f"Full dataset shape: {adata_raw.shape}")

if CELLTYPE_COL not in adata_raw.obs.columns:
    raise KeyError(
        f"Column '{CELLTYPE_COL}' not found in adata.obs. Available columns: {list(adata_raw.obs.columns)}"
    )

cell_lines = adata_raw.obs[CELLTYPE_COL].dropna().unique().tolist()
print(f"Found {len(cell_lines)} cell lines in '{CELLTYPE_COL}': {cell_lines}")


def run_pipeline_for_cell_line(adata_cl: ad.AnnData, cell_line_name: str) -> None:
    safe_cl_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(cell_line_name)).strip("_")
    print(f"\n{'='*60}\nProcessing cell line: {cell_line_name} (n={adata_cl.n_obs})\n{'='*60}")

    # --- 1. Compute HVGs from scratch on this cell line ---
    print(f"[{cell_line_name}] Computing HVGs...")
    sc.pp.highly_variable_genes(
        adata_cl,
        n_top_genes=2000,
        flavor="seurat"
    )

    hvg_mask = adata_cl.var["highly_variable"].values
    print(f"[{cell_line_name}] Number of HVGs selected: {hvg_mask.sum()}")
    adata_cl = adata_cl[:, hvg_mask].copy()

    # --- 2. Fit PCA from scratch on this cell line ---
    print(f"[{cell_line_name}] Running PCA (n_comps=50)...")
    X_cl = (
        adata_cl.X.toarray()
        if hasattr(adata_cl.X, "toarray")
        else adata_cl.X
    )

    n_comps = min(50, adata_cl.n_obs - 1, adata_cl.n_vars - 1)
    pca_model = PCA(n_components=n_comps, random_state=42)
    X_pca = pca_model.fit_transform(X_cl)

    adata_cl.obsm["X_pca"] = X_pca
    adata_cl.uns["pca"] = {
        "params": {
            "zero_center": True,
            "use_highly_variable": False,
            "n_pcs": n_comps,
            "svd_solver": "full",
        },
        "variance": pca_model.explained_variance_,
        "variance_ratio": pca_model.explained_variance_ratio_,
    }
    var_ratio = pca_model.explained_variance_ratio_
    print(f"[{cell_line_name}] PCA completed.")

    # --- 3. Select controls and specified perturbations ---
    ctrl_mask = (adata_cl.obs["perturbation"] == CONTROL_LABEL).values
    pert_mask = (adata_cl.obs["perturbation"].isin(PERTURBATIONS)).values
    combined_mask = ctrl_mask | pert_mask

    adata_sub = adata_cl[combined_mask].copy()
    n_ctrl = ctrl_mask.sum()
    n_pert_total = pert_mask.sum()

    print(f"[{cell_line_name}] Control cells: {n_ctrl}")
    print(f"[{cell_line_name}] Total perturbation cells across {PERTURBATIONS}: {n_pert_total}")
    print(f"[{cell_line_name}] Subset shape: {adata_sub.shape}")

    if adata_sub.n_obs == 0:
        print(f"[{cell_line_name}] WARNING: No control or target perturbation cells found. Skipping t-SNE.")
        return

    pcs_sub = adata_sub.obsm["X_pca"]

    # --- 4. Run t-SNE on the first 12 PCs ---
    n_tsne_pcs = min(12, n_comps)
    print(f"[{cell_line_name}] Running t-SNE on the first {n_tsne_pcs} PCs...")
    sc.tl.tsne(
        adata_sub,
        use_rep="X_pca",
        n_pcs=n_tsne_pcs,
        random_state=42
    )
    tsne_coords = adata_sub.obsm["X_tsne"]

    # --- 5. Plot t-SNE ---
    fig, ax = plt.subplots(figsize=(9, 7))

    is_ctrl_sub = (adata_sub.obs["perturbation"] == CONTROL_LABEL).values

    ax.scatter(
        tsne_coords[is_ctrl_sub, 0],
        tsne_coords[is_ctrl_sub, 1],
        c="gray",
        s=6,
        alpha=0.2,
        label=f"{CONTROL_LABEL} (n={n_ctrl})",
        rasterized=True
    )

    colors = ["blue", "green", "purple", "orange", "cyan", "magenta", "brown"]
    for i, pert in enumerate(PERTURBATIONS):
        is_pert = (adata_sub.obs["perturbation"] == pert).values
        n_pert = is_pert.sum()
        color = colors[i % len(colors)]

        ax.scatter(
            tsne_coords[is_pert, 0],
            tsne_coords[is_pert, 1],
            c=color,
            s=6,
            alpha=0.6,
            label=f"{pert} (n={n_pert})",
            rasterized=True
        )

    ax.set_title(f"t-SNE (First {n_tsne_pcs} PCs) | {DATASET} ({cell_line_name})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(
        loc="upper right",
        framealpha=0.9,
        fontsize=9,
        markerscale=2
    )

    # --- 6. Add PC1 vs PC2 inset ---
    ax_inset = ax.inset_axes([0.03, 0.62, 0.32, 0.32])

    ax_inset.scatter(
        pcs_sub[is_ctrl_sub, 0],
        pcs_sub[is_ctrl_sub, 1],
        c="gray",
        s=2,
        alpha=0.2,
        rasterized=True
    )

    for i, pert in enumerate(PERTURBATIONS):
        is_pert = (adata_sub.obs["perturbation"] == pert).values
        color = colors[i % len(colors)]

        ax_inset.scatter(
            pcs_sub[is_pert, 0],
            pcs_sub[is_pert, 1],
            c=color,
            s=2,
            alpha=0.6,
            rasterized=True
        )

    ax_inset.set_title(f"PC1 vs PC2 ({cell_line_name})", fontsize=8, fontweight="bold")
    ax_inset.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=7)
    ax_inset.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=7)
    ax_inset.tick_params(labelsize=6)

    fig.tight_layout()
    tsne_out_path = OUT_DIR / f"tsne_12pcs_pca_multi_perts_hvg_{safe_cl_name}.png"
    fig.savefig(tsne_out_path, dpi=200)
    plt.close(fig)
    print(f"[{cell_line_name}] Saved t-SNE plot: {tsne_out_path}")


for cl in cell_lines:
    adata_cl_copy = adata_raw[adata_raw.obs[CELLTYPE_COL] == cl].copy()
    run_pipeline_for_cell_line(adata_cl_copy, cl)

print("\nAll cell lines completed!")