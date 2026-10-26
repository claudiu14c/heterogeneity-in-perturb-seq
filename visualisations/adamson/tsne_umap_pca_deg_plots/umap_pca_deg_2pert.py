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
from matplotlib.lines import Line2D

print("Start")

DATASET = "adamson16"
PERTURBATIONS = ["AMIGO3", "ARHGAP22"]
CONTROL_LABEL = "control"  # Change to "control" or "non-targeting" based on dataset
KS_THRESHOLD = 0.04  # Threshold on the KS statistic D for log-transformed data

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

# --- Step 4: Run PCA on the union of DE genes ---
adata_pca = adata_combined[:, final_selected_genes].copy()
n_comps = min(50, len(final_selected_genes), adata_pca.n_obs - 1)
sc.pp.pca(adata_pca, n_comps=n_comps)
var_ratio = adata_pca.uns["pca"]["variance_ratio"]

print(f"Total variance explained by {n_comps} PCs: {var_ratio.sum() * 100:.2f}%")

# --- Step 5: Save Scree Plot ---
fig_scree, ax_scree = plt.subplots(figsize=(7, 4.5))
ax_scree.plot(range(1, len(var_ratio) + 1), var_ratio * 100, marker='o', linestyle='-', color='teal', markersize=4)
ax_scree.set_xlabel("Principal Component")
ax_scree.set_ylabel("Variance Explained (%)")
ax_scree.set_title(f"Scree Plot: {DATASET} - Multi-Perturbation Union DE")
fig_scree.tight_layout()

scree_path = OUT_DIR / f"scree_plot_pca_deg_{DATASET}_multi_pert.png"
fig_scree.savefig(scree_path, dpi=200)
plt.close(fig_scree)
print("Saved scree plot:", scree_path)

# --- Step 6: Run UMAP on PCA Space (using first 12 PCs) ---
print("Running UMAP on PCA space...")
sc.pp.neighbors(adata_pca, n_pcs=12, random_state=42)
sc.tl.umap(adata_pca, random_state=42)
umap_coords = adata_pca.obsm["X_umap"]
pca_coords = adata_pca.obsm["X_pca"]

# --- Step 7: Plot UMAP with clean inset PC1 vs PC2 plot ---
fig, ax = plt.subplots(figsize=(8, 6.5))

color_map = {
    CONTROL_LABEL: ("red", "Control"),
    PERTURBATIONS[0]: ("blue", PERTURBATIONS[0]),
    PERTURBATIONS[1]: ("green", PERTURBATIONS[1])
}

# Main UMAP scatter plot
for group_name, (color, label_text) in color_map.items():
    mask = (adata_pca.obs["perturbation"] == group_name).values
    coords = umap_coords[mask]
    ax.scatter(coords[:, 0], coords[:, 1], c=color, s=6, alpha=0.4, label=f"{label_text} (n={len(coords)})", rasterized=True)

ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title(f"UMAP (Union KS DE, D >= {KS_THRESHOLD}): {DATASET}\nMulti-Perturbation vs Control")

# Smaller inset axes for PC1 vs PC2 in the top left with semi-transparent background
ax_inset = ax.inset_axes([0.05, 0.65, 0.25, 0.25])
ax_inset.set_facecolor('white')
ax_inset.patch.set_alpha(0.85)

for group_name, (color, _) in color_map.items():
    mask = (adata_pca.obs["perturbation"] == group_name).values
    p_coords = pca_coords[mask]
    ax_inset.scatter(p_coords[:, 0], p_coords[:, 1], c=color, s=2, alpha=0.4, rasterized=True)

ax_inset.set_title(f"PC1 ({var_ratio[0]*100:.1f}%) vs PC2 ({var_ratio[1]*100:.1f}%)", fontsize=6)
ax_inset.set_xticks([])
ax_inset.set_yticks([])

fig.tight_layout()

out_path = OUT_DIR / f"umap_pca_deg_{DATASET}_multi_pert_vs_control.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print("Saved multi-perturbation UMAP plot:", out_path)

print("Done!")