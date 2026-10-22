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
from matplotlib.lines import Line2D

print("Start")

DATASET = "adamson16"
PERTURBATIONS = ["AMIGO3", "ARHGAP22"]
CONTROL_LABEL = "control"  # Change to "control" or "non-targeting" based on dataset
KS_THRESHOLD = 0.04  # Threshold on the KS statistic D for log-transformed data
N_ICA_COMPONENTS = 20  # Number of independent components (K)

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
ctrl_mask_sub = (adata_combined.obs["perturbation"] == CONTROL_LABEL).values

union_selected_genes = set()

for pert in PERTURBATIONS:
    print(f"Computing KS test for {pert} vs control...")
    pert_mask_sub = (adata_combined.obs["perturbation"] == pert).values
    
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

    # Save individual DE table to CSV
    de_csv_path = OUT_DIR / f"de_genes_{DATASET}_{pert}.csv"
    top_genes_df.to_csv(de_csv_path, index=False)
    print(f"Saved DE table: {de_csv_path}")

    # Select genes passing KS threshold
    selected_for_pert = top_genes_df.loc[top_genes_df['statistic_D'] >= KS_THRESHOLD, 'gene'].values
    if len(selected_for_pert) == 0:
        print(f"Warning: No genes passed KS threshold for {pert}. Falling back to top 500.")
        selected_for_pert = top_genes_df['gene'].head(500).values
    
    union_selected_genes.update(selected_for_pert)

# Filter union list to ensure they exist in AnnData variables
final_selected_genes = [g for g in union_selected_genes if g in adata_combined.var_names]
print(f"Total unique union DE genes selected across perturbations: {len(final_selected_genes)}")

# --- Step 4: Run ICA on the union of DE genes ---
adata_ica = adata_combined[:, final_selected_genes].copy()
n_components_ica = min(N_ICA_COMPONENTS, len(final_selected_genes), adata_ica.n_obs - 1)

print(f"Running ICA to extract {n_components_ica} independent components...")
ica_transformer = FastICA(n_components=n_components_ica, random_state=42, max_iter=1000)
ica_matrix = ica_transformer.fit_transform(
    adata_ica.X.toarray() if hasattr(adata_ica.X, "toarray") else adata_ica.X
)
adata_ica.obsm["X_ica"] = ica_matrix

# Print how many ICs were found
print(f"Number of independent components (ICs) found: {ica_matrix.shape[1]}")

# --- Step 5: Run UMAP on ICA Space ---
print("Running UMAP on ICA space...")
# Assign X_ica to X_pca slot temporarily so Scanpy's neighbor finder reads it smoothly
adata_ica.obsm["X_pca"] = adata_ica.obsm["X_ica"]
sc.pp.neighbors(adata_ica, n_pcs=n_components_ica, random_state=42)
sc.tl.umap(adata_ica, random_state=42)

umap_coords = adata_ica.obsm["X_umap"]
ica_coords = adata_ica.obsm["X_ica"]

# --- Step 6: Plot UMAP with clean inset IC1 vs IC2 plot, using high-visibility colors ---
fig, ax = plt.subplots(figsize=(8, 6.5))

# Define high-visibility color palette
# Using vibrant, colorblind-friendly colors (control is dark grey for contrast)
color_map = {
    CONTROL_LABEL: ("#333333", "Control"),  # Dark Grey
    PERTURBATIONS[0]: ("#009E73", PERTURBATIONS[0]),  # Bluish Green
    PERTURBATIONS[1]: ("#D55E00", PERTURBATIONS[1])   # Vermillion
}

# Add a subtle light grey grid to main plot for visibility
ax.grid(True, linestyle='-', linewidth=0.5, color='#F0F0F0', zorder=0)

# Main UMAP scatter plot - slightly increase dot size (s) and opacity (alpha) for visibility
for group_name, (color, label_text) in color_map.items():
    mask = (adata_ica.obs["perturbation"] == group_name).values
    coords = umap_coords[mask]
    ax.scatter(coords[:, 0], coords[:, 1], c=color, s=8, alpha=0.6, label=f"{label_text} (n={len(coords)})", rasterized=True, zorder=2)

# Legend settings for better readability
ax.legend(loc="upper right", framealpha=0.95, facecolor='white', edgecolor='#CCCCCC', fontsize=10)

# Main plot aesthetic cleanup
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title(f"UMAP on ICA (Union KS DE, D >= {KS_THRESHOLD}): {DATASET}\nMulti-Perturbation vs Control", fontsize=12)

# Inset axes for IC1 vs IC2 - adjust background for floating visibility
ax_inset = ax.inset_axes([0.05, 0.65, 0.28, 0.28])
ax_inset.set_facecolor('#FFFFFF') # Pure white background for inset
ax_inset.patch.set_alpha(1.0) # Fully opaque
ax_inset.spines[['top', 'right', 'bottom', 'left']].set_color('#AAAAAA') # Lighter spines

for group_name, (color, _) in color_map.items():
    mask = (adata_ica.obs["perturbation"] == group_name).values
    i_coords = ica_coords[mask]
    # Sightly larger dots in inset too
    ax_inset.scatter(i_coords[:, 0], i_coords[:, 1], c=color, s=3, alpha=0.5, rasterized=True)

ax_inset.set_title("IC1 vs IC2", fontsize=8, fontweight='bold', color='#555555')
ax_inset.set_xticks([])
ax_inset.set_yticks([])

fig.tight_layout()

out_path = OUT_DIR / f"umap_ica_deg_vibrant_{DATASET}_multi_pert_vs_control.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print("Saved vibrant multi-perturbation ICA-UMAP plot:", out_path)

print("Done!")