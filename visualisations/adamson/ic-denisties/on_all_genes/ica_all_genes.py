import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import FastICA
import seaborn as sns

print("Start")

DATASET = "adamson16"
PERTURBATIONS = ["STT3A", "OST4"]
CONTROL_LABEL = "control"
N_ICA_COMPONENTS = 12  # Number of independent components

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
adata_full = ad.read_h5ad(DATA_PATH)
print(f"Full dataset shape: {adata_full.shape}")

# --- Step 1: Collect masks and cell counts for control and specified perturbations ---
ctrl_mask = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
ctrl_indices = np.where(ctrl_mask)[0]
print(f"Number of control cells available: {len(ctrl_indices)}")

pert_indices_list = []
for pert in PERTURBATIONS:
    pert_mask = (adata_full.obs["perturbation"] == pert).values
    n_pert = pert_mask.sum()
    print(f"Number of perturbation cells for {pert}: {n_pert}")
    if n_pert == 0:
        print(f"Warning: No cells found for {pert}.")
    pert_indices_list.append(np.where(pert_mask)[0])

# --- Step 2: Combine control and perturbation cells (NO subsampling) ---
all_pert_indices = np.concatenate(pert_indices_list)
combined_indices = np.concatenate([ctrl_indices, all_pert_indices])

adata_combined = adata_full[combined_indices].copy()
print(f"Combined subset shape (all controls + target perturbations, all genes): {adata_combined.shape}")

# --- Step 3: Clean matrix, filter zero-variance genes, and handle NaNs ---
X_dense = adata_combined.X.toarray() if hasattr(adata_combined.X, "toarray") else adata_combined.X
X_dense = np.nan_to_num(X_dense, nan=0.0)

# Many genes have zero variance when subsetting to a few perturbations; filter them out
gene_stds = np.std(X_dense, axis=0)
valid_genes = gene_stds > 1e-6
print(f"Filtering out {(~valid_genes).sum()} zero-variance genes. Keeping {valid_genes.sum()} genes.")

adata_combined = adata_combined[:, valid_genes].copy()
X_clean = adata_combined.X.toarray() if hasattr(adata_combined.X, "toarray") else adata_combined.X
X_clean = np.nan_to_num(X_clean, nan=0.0)

# --- Step 4: Run ICA on all remaining genes ---
n_components_ica = min(N_ICA_COMPONENTS, adata_combined.n_vars, adata_combined.n_obs - 1)

print(f"Running ICA on all genes to extract {n_components_ica} independent components...")
ica_transformer = FastICA(n_components=n_components_ica, random_state=42, max_iter=1000)
ica_matrix = ica_transformer.fit_transform(X_clean)
adata_combined.obsm["X_ica"] = ica_matrix
print(f"Number of independent components (ICs) found: {ica_matrix.shape[1]}")

# --- Step 5: Plot smooth density curves for each IC across all groups ---
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

# High-visibility color palette
color_map = {
    CONTROL_LABEL: ("#333333", "Control"),  # Dark Grey
    PERTURBATIONS[0]: ("#009E73", PERTURBATIONS[0]),  # Bluish Green
    PERTURBATIONS[1]: ("#D55E00", PERTURBATIONS[1])   # Vermillion
}

for idx in range(n_components_ica):
    ax = axes[idx]
    ic_values = ica_matrix[:, idx]
    
    for group_name, (color, label_text) in color_map.items():
        mask = (adata_combined.obs["perturbation"] == group_name).values
        group_vals = ic_values[mask]
        
        # Plot smooth density estimate (KDE) line with subtle shading
        sns.kdeplot(
            group_vals, 
            ax=ax, 
            color=color, 
            label=label_text, 
            fill=True, 
            alpha=0.15, 
            linewidth=1.5
        )
    
    ax.set_title(f"IC {idx + 1}", fontsize=10, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("Density" if idx % 4 == 0 else "")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

# Hide any empty subplots if fewer than 12 components were found
for idx in range(n_components_ica, len(axes)):
    fig.delaxes(axes[idx])

fig.suptitle(f"ICA Component Density Distributions (All Genes)", fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = OUT_DIR / f"ica_all_genes_density_distributions_{PERTURBATIONS[0]}_{PERTURBATIONS[1]}.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print("Saved ICA density plot:", out_path)

print("Done!")