import os
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

print("Start")

K_NN = 20           # nearest controls averaged for the background
N_PCS_KNN = 15      # PCs of the control PCA used to find neighbours


def compute_perturbation_signature(adata, is_ctrl, k=K_NN, n_pcs=N_PCS_KNN, chunk=2000):
    """signature_i = x_i - c_i, with c_i = mean expression of the k nearest controls of cell i.
    Neighbours are found in a PCA fitted on the controls only (all cells projected into it).
    A control cell never uses itself as a neighbour."""
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    X = X.astype(np.float32)
    ctrl_idx = np.flatnonzero(is_ctrl)
    X_ctrl = X[ctrl_idx]

    pca = PCA(n_components=n_pcs, random_state=42).fit(X_ctrl)
    Z = pca.transform(X)

    # k+1 neighbours so controls can drop themselves
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z[ctrl_idx])
    _, nbr = nn.kneighbors(Z)                      # indices into ctrl_idx

    ctrl_pos = np.full(X.shape[0], -1)
    ctrl_pos[ctrl_idx] = np.arange(len(ctrl_idx))
    self_hit = nbr == ctrl_pos[:, None]            # True where a control finds itself
    drop = np.where(self_hit.any(axis=1), self_hit.argmax(axis=1), k)  # else drop the farthest
    keep = np.ones_like(nbr, dtype=bool)
    keep[np.arange(nbr.shape[0]), drop] = False
    nbr = nbr[keep].reshape(-1, k)

    sig = np.empty_like(X)
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        sig[s:e] = X[s:e] - X_ctrl[nbr[s:e]].mean(axis=1)
    return sig




DATASET = "adamson16"
PERTURBATIONS = ["STT3A", "OST4", "EIF2S1", "CARS"]
CONTROL_LABEL = "control"
PERT_COL = "perturbation"
N_HVG = 2000
N_PCS_SIG = 30      # PCs on the signatures
N_PCS_TSNE = 12     # PCs fed into t-SNE
SEED = 42

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
adata_full = ad.read_h5ad(DATA_PATH)
print(f"Full dataset shape: {adata_full.shape}")

# 1. HVGs on the whole dataset, before removing any perturbations
print("Computing global HVGs on all cells...")
sc.pp.highly_variable_genes(adata_full, n_top_genes=N_HVG, flavor="seurat")
hvg_mask = adata_full.var["highly_variable"].values
print(f"Number of global HVGs selected: {hvg_mask.sum()}")

# 2. Keep only control + target perturbations, on global HVGs
keep = adata_full.obs[PERT_COL].isin(PERTURBATIONS + [CONTROL_LABEL]).values
adata_sub = adata_full[keep, hvg_mask].copy()
del adata_full
labels = adata_sub.obs[PERT_COL].astype(str).values
is_ctrl = labels == CONTROL_LABEL
print(f"Subset shape: {adata_sub.shape} | controls: {is_ctrl.sum()}")

# 3. Signature (same function as the low-rank background script)
print("Computing signatures...")
sig = np.asarray(compute_perturbation_signature(adata_sub, is_ctrl), dtype=np.float32)
adata_sig = ad.AnnData(X=sig, obs=adata_sub.obs.copy(), var=adata_sub.var.copy())
del sig

# 4. PCA (30 PCs) on the signatures, then t-SNE on the first 12
print(f"Running PCA ({N_PCS_SIG} components) on signatures...")
sc.pp.pca(adata_sig, n_comps=N_PCS_SIG, random_state=SEED)
print(f"Running t-SNE on the first {N_PCS_TSNE} PCs...")
sc.tl.tsne(adata_sig, n_pcs=N_PCS_TSNE, random_state=SEED)
tsne = adata_sig.obsm["X_tsne"]

# 5. Plot: control gray, each perturbation its own colour
colors = dict(zip(PERTURBATIONS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(tsne[is_ctrl, 0], tsne[is_ctrl, 1], c="gray", s=5.0, alpha=0.3,
           label=f"Control (n={is_ctrl.sum()})", rasterized=True)
for p in PERTURBATIONS:
    m = labels == p
    ax.scatter(tsne[m, 0], tsne[m, 1], c=colors[p], s=5.0, alpha=0.5,
               label=f"{p} (n={m.sum()})", rasterized=True)

ax.set_title(f"{DATASET}: t-SNE of signatures ({N_PCS_TSNE} of {N_PCS_SIG} PCs)",
             fontsize=10, fontweight="bold")
ax.set_xlabel("t-SNE 1", fontsize=10)
ax.set_ylabel("t-SNE 2", fontsize=10)
ax.legend(loc="upper right", framealpha=0.9, fontsize=9, markerscale=3.0)
fig.tight_layout()

out_path = OUT_DIR / f"tsne{N_PCS_TSNE}_signature_{DATASET}_{'_'.join(PERTURBATIONS)}.png"
fig.savefig(out_path, dpi=300)
plt.close(fig)
print("Saved:", out_path)

print("Done!")