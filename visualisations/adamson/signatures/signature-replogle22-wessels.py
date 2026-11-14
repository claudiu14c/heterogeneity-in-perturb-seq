
import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

print("Start")

K_NN = 20
N_PCS_KNN = 15

DATASETS = ["replogle22", "wessels23"]
PERTURBATIONS = [
    "MED19",
    "MED22",
    "SUPT16H",
    "SUPT6H",
]
CONTROL_LABEL = "control"
PERT_COL = "perturbation"

N_HVG = 2000
N_PCS_SIG = 30
N_PCS_TSNE = 12
SEED = 42

OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT",
    "/cluster/work/boeva/virtual_cell_reasoning/data"
))


def compute_perturbation_signature(
    adata, is_ctrl, k=K_NN, n_pcs=N_PCS_KNN, chunk=2000
):
    """
    signature_i = x_i - c_i

    c_i is the mean expression of the k nearest controls.
    PCA and nearest-neighbour search are fitted independently
    within each dataset.

    A control cell never uses itself as a neighbour.
    """
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    X = X.astype(np.float32)

    ctrl_idx = np.flatnonzero(is_ctrl)
    X_ctrl = X[ctrl_idx]

    pca = PCA(n_components=n_pcs, random_state=SEED).fit(X_ctrl)
    Z = pca.transform(X)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z[ctrl_idx])
    _, nbr = nn.kneighbors(Z)

    ctrl_pos = np.full(X.shape[0], -1)
    ctrl_pos[ctrl_idx] = np.arange(len(ctrl_idx))

    self_hit = nbr == ctrl_pos[:, None]
    drop = np.where(
        self_hit.any(axis=1),
        self_hit.argmax(axis=1),
        k
    )

    keep = np.ones_like(nbr, dtype=bool)
    keep[np.arange(nbr.shape[0]), drop] = False
    nbr = nbr[keep].reshape(-1, k)

    sig = np.empty_like(X)

    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        sig[s:e] = X[s:e] - X_ctrl[nbr[s:e]].mean(axis=1)

    return sig


# ---------------------------------------------------------
# 1. Load datasets and compute HVGs independently
# ---------------------------------------------------------

adatas = {}
hvg_sets = {}

for dataset in DATASETS:
    data_path = DATA_ROOT / dataset / f"{dataset}_processed.h5ad"

    print(f"\nLoading {dataset} from: {data_path}")
    adata = ad.read_h5ad(data_path)

    print(f"{dataset} shape: {adata.shape}")

    print(f"Computing HVGs for {dataset}...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=N_HVG,
        flavor="seurat"
    )

    hvg_sets[dataset] = set(
        adata.var_names[adata.var["highly_variable"].values]
    )

    adatas[dataset] = adata

    print(
        f"{dataset}: {len(hvg_sets[dataset])} HVGs"
    )


# ---------------------------------------------------------
# 2. Find shared HVGs (intersection)
# ---------------------------------------------------------

shared_genes = set.intersection(
    *(set(adatas[d].var_names) for d in DATASETS)
)

# Union of HVGs across datasets, restricted to shared genes
shared_hvgs = sorted(
    set.union(*(hvg_sets[d] for d in DATASETS))
    & shared_genes
)

print(f"\nNumber of genes present in both datasets: {len(shared_genes)}")
print(f"Number of union HVGs present in both datasets: {len(shared_hvgs)}")

if len(shared_hvgs) == 0:
    raise ValueError("No shared HVGs between the datasets.")

# ---------------------------------------------------------
# 3. Compute signatures independently per dataset
# ---------------------------------------------------------

signature_adatas = []

for dataset in DATASETS:
    print(f"\nProcessing signatures for {dataset}...")

    adata = adatas[dataset]

    # Keep only control and target perturbations
    keep = adata.obs[PERT_COL].isin(
        PERTURBATIONS + [CONTROL_LABEL]
    ).values

    # Restrict to shared HVGs
    adata_sub = adata[keep, shared_hvgs].copy()

    labels = adata_sub.obs[PERT_COL].astype(str).values
    is_ctrl = labels == CONTROL_LABEL

    print(
        f"{dataset} subset shape: {adata_sub.shape} | "
        f"controls: {is_ctrl.sum()}"
    )

    if is_ctrl.sum() <= K_NN:
        raise ValueError(
            f"{dataset} has too few controls for k={K_NN}."
        )

    print(f"Computing signatures for {dataset}...")

    sig = compute_perturbation_signature(
        adata_sub,
        is_ctrl
    )

    adata_sig = ad.AnnData(
        X=sig,
        obs=adata_sub.obs.copy(),
        var=adata_sub.var.copy()
    )

    # Explicitly record dataset identity
    adata_sig.obs["dataset"] = dataset

    signature_adatas.append(adata_sig)

    del adata_sub, adata, sig


# ---------------------------------------------------------
# 4. Concatenate signatures across datasets
# ---------------------------------------------------------

print("\nConcatenating signature matrices...")

adata_sig_all = ad.concat(
    signature_adatas,
    axis=0,
    join="inner",
    merge="same",
    index_unique="-"
)

print(f"Combined signature shape: {adata_sig_all.shape}")

del signature_adatas, adatas


# ---------------------------------------------------------
# 5. Joint PCA on signatures from both datasets
# ---------------------------------------------------------

print(f"\nRunning joint PCA ({N_PCS_SIG} components)...")

sc.pp.pca(
    adata_sig_all,
    n_comps=N_PCS_SIG,
    random_state=SEED
)


# ---------------------------------------------------------
# 6. Joint t-SNE
# ---------------------------------------------------------

print(
    f"Running joint t-SNE on the first {N_PCS_TSNE} PCs..."
)

sc.tl.tsne(
    adata_sig_all,
    n_pcs=N_PCS_TSNE,
    random_state=SEED
)

tsne = adata_sig_all.obsm["X_tsne"]

labels = adata_sig_all.obs[PERT_COL].astype(str).values
datasets = adata_sig_all.obs["dataset"].astype(str).values

is_ctrl = labels == CONTROL_LABEL



# ---------------------------------------------------------
# 7. Plot: 20 perturbations per PNG, two plots per perturbation
# ---------------------------------------------------------

import matplotlib.colors as mcolors

PLOTS_PER_FILE = 20
N_COLS = 4
N_ROWS = 10  # 2 rows per perturbation

# Automatic perturbation colours
cmap = plt.get_cmap("tab20", max(len(PERTURBATIONS), 1))

colors = {
    p: mcolors.to_hex(cmap(i))
    for i, p in enumerate(PERTURBATIONS)
}

markers = {
    DATASETS[0]: "o",
    DATASETS[1]: "^"
}

available_perturbations = [
    p for p in PERTURBATIONS
    if p != CONTROL_LABEL
    and all(np.any((datasets == ds) & (labels == p))
            for ds in DATASETS)
]

for batch_start in range(0, len(available_perturbations), PLOTS_PER_FILE):

    batch = available_perturbations[
        batch_start:batch_start + PLOTS_PER_FILE
    ]

    batch_number = batch_start // PLOTS_PER_FILE + 1

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(24, 32),
        squeeze=False
    )

    for i, perturbation in enumerate(batch):

        # Two vertically stacked plots per perturbation
        for j, dataset in enumerate(DATASETS):

            ax = axes[2 * (i % 10) + j, i // 10]

            # Plot controls from BOTH datasets
            for ctrl_dataset in DATASETS:

                ctrl_mask = (
                    (datasets == ctrl_dataset) &
                    (labels == CONTROL_LABEL)
                )

                if ctrl_mask.any():
                    ax.scatter(
                        tsne[ctrl_mask, 0],
                        tsne[ctrl_mask, 1],
                        c="lightgray",
                        marker=markers[ctrl_dataset],
                        s=5,
                        alpha=0.2,
                        rasterized=True
                    )

            # Plot perturbation from the selected dataset only
            pert_mask = (
                (datasets == dataset) &
                (labels == perturbation)
            )

            ax.scatter(
                tsne[pert_mask, 0],
                tsne[pert_mask, 1],
                c=colors[perturbation],
                marker=markers[dataset],
                s=6,
                alpha=0.6,
                rasterized=True
            )

            ax.set_title(
                f"{perturbation} | {dataset} "
                f"(n={pert_mask.sum()})",
                fontsize=9,
                fontweight="bold"
            )

            ax.set_xlabel("t-SNE 1", fontsize=8)
            ax.set_ylabel("t-SNE 2", fontsize=8)
            ax.tick_params(labelsize=7)

    # Hide unused axes
    used_axes = 2 * len(batch)

    for ax in axes.flat:
        if not ax.collections:
            ax.set_visible(False)

    # Compact shared legend
    handles = [
        plt.Line2D(
            [0], [0],
            marker=markers[ds],
            color="lightgray",
            linestyle="None",
            markersize=5,
            label=f"Control - {ds}"
        )
        for ds in DATASETS
    ]

    for ds in DATASETS:
        handles.append(
            plt.Line2D(
                [0], [0],
                marker=markers[ds],
                color="black",
                linestyle="None",
                markersize=5,
                label=f"Perturbation - {ds}"
            )
        )

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=7,
        markerscale=0.8,
        framealpha=0.8
    )

    fig.suptitle(
        f"Joint t-SNE perturbation signatures | "
        f"{', '.join(DATASETS)} | Batch {batch_number}",
        fontsize=14,
        fontweight="bold",
        y=1.01
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = OUT_DIR / (
        f"joint_tsne_separate_datasets_batch_{batch_number:02d}_"
        f"{'_'.join(DATASETS)}.png"
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")

print(
    f"\nDone! Generated "
    f"{(len(available_perturbations) + 19) // 20} PNG files "
    f"for {len(available_perturbations)} perturbations."
)
