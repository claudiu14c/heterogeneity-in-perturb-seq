
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

DATASETS = ["adamson16", "replogle22"]
PERTURBATIONS = ["STT3A", "OST4", "EIF2S1", "CARS"]
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

shared_hvgs = sorted(
    set.intersection(*(hvg_sets[d] for d in DATASETS))
)

print(f"\nNumber of shared HVGs: {len(shared_hvgs)}")

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
# 7. Plot: perturbation colours, dataset marker styles
# ---------------------------------------------------------

colors = {
    "STT3A": "#0072B2",
    "OST4": "#D55E00",
    "EIF2S1": "#009E73",
    "CARS": "#CC79A7"
}

markers = {
    "adamson16": "o",
    "replogle22": "^"
}

fig, ax = plt.subplots(figsize=(10, 8))

for dataset in DATASETS:
    for perturbation in [CONTROL_LABEL] + PERTURBATIONS:

        m = (datasets == dataset) & (labels == perturbation)

        if not m.any():
            continue

        color = "gray" if perturbation == CONTROL_LABEL else colors[perturbation]

        ax.scatter(
            tsne[m, 0],
            tsne[m, 1],
            c=color,
            marker=markers[dataset],
            s=5.0,
            alpha=0.3 if perturbation == CONTROL_LABEL else 0.5,
            label=f"{dataset} - {perturbation} (n={m.sum()})",
            rasterized=True
        )

ax.set_title(
    f"Joint t-SNE of perturbation signatures\n"
    f"{', '.join(DATASETS)} | {N_PCS_TSNE} of {N_PCS_SIG} PCs",
    fontsize=11,
    fontweight="bold"
)

ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")

ax.legend(
    loc="best",
    framealpha=0.9,
    fontsize=8,
    markerscale=3.0
)

fig.tight_layout()

out_path = OUT_DIR / (
    f"joint_tsne{N_PCS_TSNE}_signature_"
    f"{'_'.join(DATASETS)}_"
    f"{'_'.join(PERTURBATIONS)}.png"
)

fig.savefig(out_path, dpi=300)
plt.close(fig)

print("Saved:", out_path)

# Optionally save the joint embedding and PCA
embedding_path = OUT_DIR / (
    f"joint_signatures_{'_'.join(DATASETS)}.h5ad"
)

adata_sig_all.write_h5ad(embedding_path)

print("Saved:", embedding_path)
print("\nDone!")
