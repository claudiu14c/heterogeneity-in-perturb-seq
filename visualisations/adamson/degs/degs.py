import os
from pathlib import Path
import anndata as ad
import matplotlib
matplotlib.use("Agg")  # render straight to file
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc

DATASET = "adamson16"
CONTROL_LABEL = "control"  # Adjust if your dataset uses "non-targeting"

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
adata = ad.read_h5ad(DATA_PATH)

# Get all unique perturbations (excluding control)
perturbations = [p for p in adata.obs["perturbation"].unique() if p != CONTROL_LABEL]
ctrl_mask = (adata.obs["perturbation"] == CONTROL_LABEL).values
ctrl_indices = np.where(ctrl_mask)[0]

deg_counts = []

print(f"Calculating DEG counts for {len(perturbations)} perturbations...")
for pert in perturbations:
    pert_mask = (adata.obs["perturbation"] == pert).values
    if pert_mask.sum() < 5:  # Skip if too few cells
        continue
        
    sub_indices = np.where(pert_mask | ctrl_mask)[0]
    adata_sub = adata[sub_indices].copy()
    
    try:
        sc.tl.rank_genes_groups(
            adata_sub, 
            groupby="perturbation", 
            groups=[pert], 
            reference=CONTROL_LABEL, 
            method="wilcoxon"
        )
        de_df = sc.get.rank_genes_groups_df(adata_sub, group=pert)
        
        # Filter by significance thresholds
        sig_de = de_df[
            (de_df["pvals_adj"] < 0.05) & (de_df["logfoldchanges"].abs() > 0.1)
        ]
        deg_counts.append(len(sig_de))
    except Exception as e:
        print(f"Skipping {pert}: {e}")

# Define uniform bins of 100 each, spanning up to the max observed DEG count
max_deg = max(deg_counts) if deg_counts else 100
bin_width = 100
max_edge = int(np.ceil((max_deg + 1) / bin_width) * bin_width)

bin_edges = list(range(0, max_edge + bin_width, bin_width))
bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges) - 1)]

# Bin the DEG counts using pandas
counts_series = pd.Series(deg_counts)
binned = pd.cut(counts_series, bins=bin_edges, labels=bin_labels, right=False, include_lowest=True)
binned_counts = binned.value_counts().reindex(bin_labels).fillna(0)

# Plot bar chart
fig, ax = plt.subplots(figsize=(12, 5))
binned_counts.plot(kind="bar", color="steelblue", edgecolor="black", alpha=0.8, ax=ax)

ax.set_title(f"Distribution of DEG Counts per Perturbation (100-gene bins, {DATASET})", fontsize=11)
ax.set_xlabel("Number of DEGs Interval", fontsize=10)
ax.set_ylabel("Number of Perturbations", fontsize=10)
plt.xticks(rotation=45, ha="right")

fig.tight_layout()
out_path = OUT_DIR / f"deg_binned_100_distribution_{DATASET}.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)

print(f"Saved binned distribution plot to: {out_path}")
print("Done!")