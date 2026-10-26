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

# --- Step 1: Collect masks and cell counts for both perturbations and control ---
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

# --- Step 2: Combine all control and perturbation cells (NO subsampling) ---
all_pert_indices = np.concatenate(pert_indices_list)
combined_indices = np.concatenate([ctrl_indices, all_pert_indices])

adata_combined = adata_full[combined_indices].copy()
print(f"Combined subset shape (all controls + all specified perturbations): {adata_combined.shape}")

# --- Step 3: Compute DEGs using Wilcoxon rank-sum test for each perturbation vs control, take union ---
union_selected_genes = set()

for pert in PERTURBATIONS:
    print(f"Computing Wilcoxon rank-sum DEGs for {pert} vs control...")
    pert_mask_sub = (adata_combined.obs["perturbation"] == pert).values
    ctrl_mask_sub = (adata_combined.obs["perturbation"] == CONTROL_LABEL).values
    
    # Subset to this perturbation + control for DE testing
    sub_indices = np.where(pert_mask_sub | ctrl_mask_sub)[0]
    adata_sub = adata_combined[sub_indices].copy()
    
    sc.tl.rank_genes_groups(
        adata_sub, 
        groupby="perturbation", 
        groups=[pert], 
        reference=CONTROL_LABEL, 
        method="wilcoxon"
    )
    
    de_df = sc.get.rank_genes_groups_df(adata_sub, group=pert)
    
    # Filter by adjusted p-value < 0.05 AND absolute log fold change > 0.1
    sig_de_df = de_df[
        (de_df["pvals_adj"] < 0.05) & (de_df["logfoldchanges"].abs() > 0.1)
    ]
    genes = sig_de_df["names"].values
    
    if len(genes) == 0:
        print(f"Warning: No significant DEGs found for {pert}. Using top 100 unadjusted.")
        genes = de_df["names"].values[:100]
        
    print(f"-> Significant DEGs for {pert}: {len(genes)}")
    union_selected_genes.update(genes)

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

fig.suptitle(f"ICA Component Density Distributions (Wilcoxon Union DEGs)", fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = OUT_DIR / f"ica_on_deg_density_distributions_{PERTURBATIONS[0]}_{PERTURBATIONS[1]}.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print("Saved ICA density plot:", out_path)

print("Done!")