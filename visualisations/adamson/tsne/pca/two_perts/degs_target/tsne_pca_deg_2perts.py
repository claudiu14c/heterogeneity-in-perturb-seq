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
CONTROL_LABEL = "control"

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

# 2. Prepare Combined Subset using the Union of DEGs
pert_mask_all = adata_full.obs["perturbation"].isin(PERTURBATIONS).values
pert_indices = np.where(pert_mask_all)[0]
combined_indices = np.concatenate([pert_indices, ctrl_indices])

adata_combined = adata_full[combined_indices, :][:, union_de_genes].copy()

print("Running PCA (12 components) on combined subset using union of DEGs...")
sc.pp.pca(adata_combined, n_comps=12, random_state=42)
pcs_combined = adata_combined.obsm["X_pca"]
var_ratio_combined = adata_combined.uns["pca"]["variance_ratio"]

print("Running t-SNE on the first 12 PCs...")
sc.tl.tsne(adata_combined, n_pcs=12, random_state=42)
tsne_coords = adata_combined.obsm["X_tsne"]

# 3. Plotting Single Panel: t-SNE (on 12 PCs) with PCA inset
fig, ax = plt.subplots(figsize=(8, 6))

is_ctrl_comb = (adata_combined.obs["perturbation"] == CONTROL_LABEL).values

# Main scatter plot (t-SNE)
ax.scatter(
    tsne_coords[is_ctrl_comb, 0], tsne_coords[is_ctrl_comb, 1], 
    c="gray", s=6, alpha=0.2, 
    label=f"{CONTROL_LABEL} (n={len(ctrl_indices)})", rasterized=True
)

colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]
for i, pert in enumerate(PERTURBATIONS):
    is_pert = (adata_combined.obs["perturbation"] == pert).values
    n_pert = is_pert.sum()
    color = colors[i % len(colors)]
    
    ax.scatter(
        tsne_coords[is_pert, 0], tsne_coords[is_pert, 1], 
        c=color, s=6, alpha=0.5, 
        label=f"{pert} (n={n_pert})", rasterized=True
    )

ax.set_title("t-SNE (on first 12 PCs, Union DEGs)", fontsize=11, fontweight='bold')
ax.set_xlabel("t-SNE 1", fontsize=10)
ax.set_ylabel("t-SNE 2", fontsize=10)
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

# Subplot inset in the top left corner (PC1 vs PC2)
ax_inset = ax.inset_axes([0.03, 0.62, 0.32, 0.32])

ax_inset.scatter(
    pcs_combined[is_ctrl_comb, 0], pcs_combined[is_ctrl_comb, 1], 
    c="gray", s=3, alpha=0.2, rasterized=True
)
for i, pert in enumerate(PERTURBATIONS):
    is_pert = (adata_combined.obs["perturbation"] == pert).values
    color = colors[i % len(colors)]
    ax_inset.scatter(
        pcs_combined[is_pert, 0], pcs_combined[is_pert, 1], 
        c=color, s=3, alpha=0.5, rasterized=True
    )

ax_inset.set_title("PC1 vs PC2", fontsize=8, fontweight='bold')
ax_inset.set_xlabel(f"PC1 ({var_ratio_combined[0]*100:.1f}%)", fontsize=7)
ax_inset.set_ylabel(f"PC2 ({var_ratio_combined[1]*100:.1f}%)", fontsize=7)
ax_inset.tick_params(labelsize=6)

# Format DEG summary text for the figure title
deg_summary_parts = [f"{p}: {c} DEGs" for p, c in deg_counts.items()]
deg_text = " | ".join(deg_summary_parts) + f" | Union Total: {total_union_size} DEGs"

fig.suptitle(f"Dataset: {DATASET} | Combined Perturbations\n{deg_text}", fontsize=10)
fig.tight_layout()

out_path = OUT_DIR / f"tsne_pca_deg_on_2_perts_{DATASET}.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print("Saved:", out_path)

print("Done!")