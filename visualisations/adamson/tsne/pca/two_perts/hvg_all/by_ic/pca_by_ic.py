import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scanpy as sc
from sklearn.decomposition import FastICA

print("Start")

DATASET = "adamson16"
PERTURBATIONS = ["STT3A", "OST4"]
CONTROL_LABEL = "control"
N_ICA_COMPONENTS = 12
N_TOP_HVGS = 2000

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
adata_full = ad.read_h5ad(DATA_PATH)
print(f"Full dataset shape: {adata_full.shape}")

# 1. Compute Global Highly Variable Genes (HVGs) on all cells in the dataset
print("Computing Global Highly Variable Genes (HVGs) on all cells...")
sc.pp.highly_variable_genes(adata_full, n_top_genes=N_TOP_HVGS, flavor="seurat")
hvg_mask = adata_full.var["highly_variable"].values
print(f"Number of global HVGs selected: {hvg_mask.sum()}")

# 2. Subset dataset to control + the 2 chosen perturbations only, keeping only global HVGs
target_groups = PERTURBATIONS + [CONTROL_LABEL]
subset_mask = adata_full.obs["perturbation"].isin(target_groups).values

adata_sub = adata_full[subset_mask, hvg_mask].copy()
print(f"Subset shape (Control + {PERTURBATIONS} on Global HVGs): {adata_sub.shape}")

# 3. Run PCA (12 components) and t-SNE on this subset
print("Running PCA (12 components) on subset...")
sc.pp.pca(adata_sub, n_comps=12, random_state=42)

print("Running t-SNE on the first 12 PCs...")
sc.tl.tsne(adata_sub, n_pcs=12, random_state=42)
tsne_coords = adata_sub.obsm["X_tsne"]

# 4. Run FastICA on the subset (global HVGs) to get 12 independent components
print(f"Running FastICA to extract {N_ICA_COMPONENTS} independent components...")
X_dense = adata_sub.X.toarray() if hasattr(adata_sub.X, "toarray") else adata_sub.X
ica_transformer = FastICA(n_components=N_ICA_COMPONENTS, random_state=42, max_iter=1000)
ica_matrix = ica_transformer.fit_transform(X_dense)
print(f"ICA matrix shape: {ica_matrix.shape}")

# 5. Plotting: 3x4 grid of t-SNE subplots colored by each IC score
fig, axes = plt.subplots(3, 4, figsize=(18, 13))
axes = axes.flatten()

for idx in range(N_ICA_COMPONENTS):
    ax = axes[idx]
    ic_values = ica_matrix[:, idx]
    
    # Set symmetric limits around zero so that 0 maps exactly to white in coolwarm
    max_val = np.max(np.abs(ic_values))
    if max_val == 0:
        max_val = 1.0
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-max_val, vmax=max_val)
    
    sc_plot = ax.scatter(
        tsne_coords[:, 0], tsne_coords[:, 1],
        c=ic_values, cmap="coolwarm", norm=norm,
        s=5.0, alpha=0.6, rasterized=True
    )
    
    ax.set_title(f"IC {idx + 1}", fontsize=10, fontweight='bold')
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.tick_params(labelsize=7)
    
    fig.colorbar(sc_plot, ax=ax, shrink=0.8, pad=0.02)

fig.suptitle(f"t-SNE Colored by Independent Components (Dataset: {DATASET} | {PERTURBATIONS[0]} & {PERTURBATIONS[1]})", fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = OUT_DIR / f"tsne12_ica_colored_{DATASET}_{PERTURBATIONS[0]}_{PERTURBATIONS[1]}.png"
fig.savefig(out_path, dpi=300)
plt.close(fig)
print("Saved:", out_path)

print("Done!")