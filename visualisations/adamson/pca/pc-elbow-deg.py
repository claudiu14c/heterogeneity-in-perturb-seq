import os
from pathlib import Path
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ks_2samp

print("Start PCA Elbow Plot Generation")

DATASET = "adamson16"
PERTURBATIONS = ["STT3A", "OST4"]
CONTROL_LABEL = "control"
KS_THRESHOLD = 0.04

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
adata_full = ad.read_h5ad(DATA_PATH)

# --- Step 1: Collect masks and cell counts for STT3A and OST4 ---
pert_masks = {}
n_perts = {}
for pert in PERTURBATIONS:
    mask = (adata_full.obs["perturbation"] == pert).values
    pert_masks[pert] = mask
    n_perts[pert] = mask.sum()
    print(f"Number of perturbation cells for {pert}: {n_perts[pert]}")

ctrl_mask = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
n_ctrl_total = ctrl_mask.sum()

# --- Step 2: Downsample control cells to average size of the perturbations ---
avg_pert_size = int(np.mean(list(n_perts.values())))
print(f"Target downsampled control size: {avg_pert_size}")

ctrl_indices = np.where(ctrl_mask)[0]
np.random.seed(42)
if n_ctrl_total > avg_pert_size:
    sampled_ctrl_indices = np.random.choice(ctrl_indices, size=avg_pert_size, replace=False)
else:
    sampled_ctrl_indices = ctrl_indices

all_pert_indices = np.concatenate([np.where(pert_masks[pert])[0] for pert in PERTURBATIONS])
combined_indices = np.concatenate([sampled_ctrl_indices, all_pert_indices])
adata_combined = adata_full[combined_indices].copy()

# --- Step 3: Compute union DEGs using KS test ---
X_data = adata_combined.X.toarray() if hasattr(adata_combined.X, "toarray") else adata_combined.X
gene_names = adata_combined.var_names
ctrl_mask_sub = (adata_combined.obs["perturbation"] == CONTROL_LABEL).values

union_selected_genes = set()

for pert in PERTURBATIONS:
    print(f"Computing KS test for {pert} vs control...")
    pert_mask_sub = (adata_combined.obs["perturbation"] == pert).values
    
    ks_results = []
    for i, gene in enumerate(gene_names):
        res = ks_2samp(X_data[pert_mask_sub, i], X_data[ctrl_mask_sub, i])
        ks_results.append({'gene': gene, 'statistic_D': res.statistic})

    top_genes_df = pd.DataFrame(ks_results).sort_values(by='statistic_D', ascending=False)
    selected_for_pert = top_genes_df.loc[top_genes_df['statistic_D'] >= KS_THRESHOLD, 'gene'].values
    if len(selected_for_pert) == 0:
        selected_for_pert = top_genes_df['gene'].head(500).values
    
    union_selected_genes.update(selected_for_pert)

final_selected_genes = [g for g in union_selected_genes if g in adata_combined.var_names]
print(f"Total union DE genes selected across {PERTURBATIONS}: {len(final_selected_genes)}")

# --- Step 4: Run PCA on the union of DE genes ---
adata_pca = adata_combined[:, final_selected_genes].copy()
n_comps = min(50, adata_pca.n_obs - 1, adata_pca.n_vars)

print(f"Running PCA with {n_comps} components...")
sc.pp.pca(adata_pca, n_comps=n_comps, random_state=42)

# Extract explained variance ratio per principal component
var_ratio = adata_pca.uns["pca"]["variance_ratio"]

# --- Step 5: Plot and save PCA Elbow Plot ---
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    range(1, len(var_ratio) + 1), 
    var_ratio * 100, 
    marker='o', 
    color='#009E73', 
    linestyle='-', 
    linewidth=2, 
    markersize=5
)

ax.set_title(f"PCA Variance (Union KS DEGs, D >= {KS_THRESHOLD})\nPerturbations: {', '.join(PERTURBATIONS)}", fontsize=11, fontweight='bold')
ax.set_xlabel("Principal Component", fontsize=10)
ax.set_ylabel("Explained Variance (%)", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)

fig.tight_layout()

out_path = OUT_DIR / f"pc_elbow_{DATASET}_combined_perturbations_union_degs.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)

print(f"Saved PCA elbow plot to: {out_path}")
print("Done!")