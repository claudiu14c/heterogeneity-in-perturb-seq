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
N_TOP_HVGS = 2000      # Number of top highly variable genes to select globally

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
adata_full = ad.read_h5ad(DATA_PATH)
print(f"Full dataset shape: {adata_full.shape}")

# --- Step 1: Compute Global HVGs across ALL cells in the dataset ---
print(f"Computing top {N_TOP_HVGS} Highly Variable Genes globally on all cells...")
sc.pp.highly_variable_genes(adata_full, n_top_genes=N_TOP_HVGS, flavor="seurat")
hvg_mask = adata_full.var["highly_variable"].values
print(f"Number of global HVGs selected: {hvg_mask.sum()}")

# --- Step 2: Collect masks and cell counts for control and specified perturbations ---
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

# --- Step 3: Combine control and perturbation cells (NO subsampling) and subset to global HVGs ---
all_pert_indices = np.concatenate(pert_indices_list)
combined_indices = np.concatenate([ctrl_indices, all_pert_indices])

adata_ica = adata_full[combined_indices, hvg_mask].copy()
print(f"Combined subset shape (all controls + target perturbations, global HVGs): {adata_ica.shape}")

# --- Step 4: Run ICA on the global HVGs with 12 components ---
n_components_ica = min(N_ICA_COMPONENTS, adata_ica.n_vars, adata_ica.n_obs - 1)

print(f"Running ICA on global HVGs to extract {n_components_ica} independent components...")
X_dense = adata_ica.X.toarray() if hasattr(adata_ica.X, "toarray") else adata_ica.X

ica_transformer = FastICA(n_components=n_components_ica, random_state=42, max_iter=1000)
ica_matrix = ica_transformer.fit_transform(X_dense)
adata_ica.obsm["X_ica"] = ica_matrix
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
        mask = (adata_ica.obs["perturbation"] == group_name).values
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

fig.suptitle(f"ICA Component Density Distributions (Global HVGs, Top {N_TOP_HVGS})", fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = OUT_DIR / f"ica_global_hvgs_density_distributions_{PERTURBATIONS[0]}_{PERTURBATIONS[1]}.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print("Saved ICA density plot:", out_path)

print("Done!")