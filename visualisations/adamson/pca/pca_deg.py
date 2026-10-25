import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

print("Start")

DATASET = "adamson16"
PERTURBATIONS = ["STT3A", "OST4"]
CONTROL_LABEL = "control"  # Change to "control" or "non-targeting" based on dataset

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
adata_full = ad.read_h5ad(DATA_PATH)
print(f"Full dataset shape: {adata_full.shape}")

# 1. Compute DEGs for each perturbation individually and take the union
union_de_genes = set()
ctrl_mask = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
ctrl_indices = np.where(ctrl_mask)[0]

deg_counts = {}

for pert in PERTURBATIONS:
    print(f"\n--- Finding DEGs for perturbation: {pert} vs {CONTROL_LABEL} ---")
    pert_mask = (adata_full.obs["perturbation"] == pert).values
    
    n_pert = pert_mask.sum()
    print(f"Number of cells for {pert}: {n_pert}")

    if n_pert == 0:
        print(f"Warning: No cells found for {pert}. Skipping.")
        deg_counts[pert] = 0
        continue
        
    # Subset to this perturbation + control for DE testing
    sub_indices = np.where(pert_mask | ctrl_mask)[0]
    adata_sub = adata_full[sub_indices].copy()
    
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
        print(f"Warning: No significant DEGs found for {pert} with p_adj < 0.05 & |logFC| > 0.1. Using top 100 unadjusted.")
        genes = de_df["names"].values[:100]
        
    deg_counts[pert] = len(genes)
    print(f"-> Significant DEGs for {pert}: {len(genes)}")
    union_de_genes.update(genes)

# Filter to genes actually present in the dataset
union_de_genes = [g for g in union_de_genes if g in adata_full.var_names]
total_union_size = len(union_de_genes)

print("\n===============================")
print("DEGs Summary (p_adj < 0.05, |logFC| > 0.1):")
for pert, count in deg_counts.items():
    print(f"  - {pert}: {count} DEGs")
print(f"  - Total Unique Union DEGs: {total_union_size}")
print("===============================\n")

# 2. Prepare Subsets using the Union of DEGs
pert_mask_all = adata_full.obs["perturbation"].isin(PERTURBATIONS).values
pert_indices = np.where(pert_mask_all)[0]
combined_indices = np.concatenate([pert_indices, ctrl_indices])

# Controls-only subset on union DEGs
adata_ctrl = adata_full[ctrl_indices, :][:, union_de_genes].copy()
sc.pp.pca(adata_ctrl, n_comps=2)
pcs_ctrl_only = adata_ctrl.obsm["X_pca"]
var_ratio_ctrl = adata_ctrl.uns["pca"]["variance_ratio"]

# Combined subset (Controls + All Perturbations) on union DEGs
adata_combined = adata_full[combined_indices, :][:, union_de_genes].copy()
print("Running PCA on combined subset using union of DEGs...")
sc.pp.pca(adata_combined, n_comps=2)
pcs_combined = adata_combined.obsm["X_pca"]
var_ratio_combined = adata_combined.uns["pca"]["variance_ratio"]

# 3. Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Subplot 1: Controls only ---
axes[0].scatter(
    pcs_ctrl_only[:, 0], pcs_ctrl_only[:, 1], 
    c="red", s=6, alpha=0.4, 
    label=f"{CONTROL_LABEL} (n={len(ctrl_indices)})", rasterized=True
)
axes[0].set_title("PCA (Union DEGs): Controls Only")
axes[0].set_xlabel(f"PC1 ({var_ratio_ctrl[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({var_ratio_ctrl[1]*100:.1f}%)")
axes[0].legend(loc="upper right", framealpha=0.9, fontsize=9)

# --- Subplot 2: Controls + All Perturbations on the same plot ---
is_ctrl_comb = (adata_combined.obs["perturbation"] == CONTROL_LABEL).values
axes[1].scatter(
    pcs_combined[is_ctrl_comb, 0], pcs_combined[is_ctrl_comb, 1], 
    c="gray", s=6, alpha=0.2, 
    label=f"{CONTROL_LABEL} (n={len(ctrl_indices)})", rasterized=True
)

colors = ["blue", "green", "purple", "orange", "cyan"]
for i, pert in enumerate(PERTURBATIONS):
    is_pert = (adata_combined.obs["perturbation"] == pert).values
    n_pert = is_pert.sum()
    color = colors[i % len(colors)]
    
    axes[1].scatter(
        pcs_combined[is_pert, 0], pcs_combined[is_pert, 1], 
        c=color, s=6, alpha=0.5, 
        label=f"{pert} (n={n_pert})", rasterized=True
    )

axes[1].set_title("PCA (Union DEGs): Combined Perturbations vs All Controls")
axes[1].set_xlabel(f"PC1 ({var_ratio_combined[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({var_ratio_combined[1]*100:.1f}%)")
axes[1].legend(loc="upper right", framealpha=0.9, fontsize=9)

# Format DEG summary text for the figure title
deg_summary_parts = [f"{p}: {c} DEGs" for p, c in deg_counts.items()]
deg_text = " | ".join(deg_summary_parts) + f" | Union Total: {total_union_size} DEGs"

fig.suptitle(f"Dataset: {DATASET} | Combined Perturbations\n{deg_text}", fontsize=10)
fig.tight_layout()

out_path = OUT_DIR / f"pca12_{DATASET}_combined_perturbations_union_degs.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print("Saved:", out_path)

print("Done!")