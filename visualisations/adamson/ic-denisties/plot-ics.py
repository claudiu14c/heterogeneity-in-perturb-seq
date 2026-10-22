import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ks_2samp
from sklearn.decomposition import FastICA
import seaborn as sns

print("Start")

DATASET = "adamson16"
PERTURBATIONS = ["AMIGO3", "ARHGAP22"]
CONTROL_LABEL = "control"  # Change to "control" or "non-targeting" based on dataset
KS_THRESHOLD = 0.04  # Threshold on the KS statistic D for log-transformed data
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

# --- Step 1: Collect masks and cell counts for both perturbations ---
pert_masks = {}
n_perts = {}
for pert in PERTURBATIONS:
    mask = (adata_full.obs["perturbation"] == pert).values
    pert_masks[pert] = mask
    n_perts[pert] = mask.sum()
    print(f"Number of perturbation cells for {pert}: {n_perts[pert]}")
    if n_perts[pert] == 0:
        print(f"Warning: No cells found for {pert}.")

ctrl_mask = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
n_ctrl_total = ctrl_mask.sum()
print(f"Number of control cells available: {n_ctrl_total}")

# --- Step 2: Downsample control cells to average size of the perturbations ---
avg_pert_size = int(np.mean(list(n_perts.values())))
print(f"Target downsampled control size (average of perturbation sizes): {avg_pert_size}")

ctrl_indices = np.where(ctrl_mask)[0]
np.random.seed(42)  # Set seed for reproducibility
if n_ctrl_total > avg_pert_size:
    sampled_ctrl_indices = np.random.choice(ctrl_indices, size=avg_pert_size, replace=False)
else:
    sampled_ctrl_indices = ctrl_indices

# Combine control and all perturbation cell indices
all_pert_indices = np.concatenate([np.where(pert_masks[pert])[0] for pert in PERTURBATIONS])
combined_indices = np.concatenate([sampled_ctrl_indices, all_pert_indices])

adata_combined = adata_full[combined_indices].copy()
print(f"Combined balanced subset shape (control + all perturbations): {adata_combined.shape}")

# --- Step 3: Compute DEGs using KS test for each perturbation vs control, then take union ---
X_data = adata_combined.X.toarray() if hasattr(adata_combined.X, "toarray") else adata_combined.X
gene_names = adata_combined.var_names

union_selected_genes = set()

for pert in PERTURBATIONS:
    print(f"Computing KS test for {pert} vs control...")
    pert_mask_sub = (adata_combined.obs["perturbation"] == pert).values
    ctrl_mask_sub = (adata_combined.obs["perturbation"] == CONTROL_LABEL).values
    
    ks_results = []
    for i, gene in enumerate(gene_names):
        expr_pert = X_data[pert_mask_sub, i]
        expr_ctrl = X_data[ctrl_mask_sub, i]
        res = ks_2samp(expr_pert, expr_ctrl)
        ks_results.append({
            'gene': gene,
            'statistic_D': res.statistic,
            'pvals': res.pvalue
        })

    top_genes_df = pd.DataFrame(ks_results)
    top_genes_df = top_genes_df.sort_values(by='statistic_D', ascending=False).reset_index(drop=True)

    # Select genes passing KS threshold
    selected_for_pert = top_genes_df.loc[top_genes_df['statistic_D'] >= KS_THRESHOLD, 'gene'].values
    if len(selected_for_pert) == 0:
        print(f"Warning: No genes passed KS threshold for {pert}. Falling back to top 500.")
        selected_for_pert = top_genes_df['gene'].head(500).values
    
    union_selected_genes.update(selected_for_pert)

# Filter union list to ensure they exist in AnnData variables
final_selected_genes = [g for g in union_selected_genes if g in adata_combined.var_names]
print(f"Total unique union DE genes selected across perturbations: {len(final_selected_genes)}")

# --- Step 4: Run ICA on the union of DE genes with 12 components ---
adata_ica = adata_combined[:, final_selected_genes].copy()
n_components_ica = min(N_ICA_COMPONENTS, len(final_selected_genes), adata_ica.n_obs - 1)

print(f"Running ICA to extract {n_components_ica} independent components...")
ica_transformer = FastICA(n_components=n_components_ica, random_state=42, max_iter=1000)
ica_matrix = ica_transformer.fit_transform(
    adata_ica.X.toarray() if hasattr(adata_ica.X, "toarray") else adata_ica.X
)
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

fig.suptitle(f"ICA Component Density Distributions (Union KS DE, D >= {KS_THRESHOLD})", fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = OUT_DIR / f"ica_density_distributions_{DATASET}.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print("Saved ICA density plot:", out_path)

print("Done!")