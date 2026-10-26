import os
from pathlib import Path
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
from sklearn.linear_model import Ridge

print("Start Unreliable Perturbations PS Score Analysis (No Downsampling)")

DATASET = "adamson16"
CONTROL_LABEL = "control"
BATCH_SIZE = 15  # 15 subplots per image (3 rows x 5 columns)

SPLIT_CSV_URL = "reliability_adamson.csv"

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
adata_full = ad.read_h5ad(DATA_PATH)

# Load split metadata and filter for 'unreliable' perturbations
print(f"Loading split CSV from: {SPLIT_CSV_URL}")
try:
    split_df = pd.read_csv(SPLIT_CSV_URL)
    unreliable_perts = split_df.loc[split_df["quality_label"] == "unreliable", "perturbation"].astype(str).tolist()
except Exception as e:
    print(f"Could not load split CSV directly ({e}). Using fallback list of unreliable perturbations.")
    unreliable_perts = [
        "AMIGO3", "ATF4", "ERN1", "GMPPB", "GNPNAT1", "HYOU1", 
        "KCTD16", "MANF", "PSMA1", "PSMD12", "PTDSS1", "SEC63", 
        "SLC35B1", "STT3A", "TTI2" #, "XBP1" ignore the last one so that all first 15 fit on one plot
    ]

# Keep only those present in the dataset
available_perts = [p for p in unreliable_perts if p in adata_full.obs["perturbation"].astype(str).unique()]
print(f"Found {len(available_perts)} unreliable perturbations in dataset: {available_perts}")

ctrl_mask_full = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
ctrl_indices_full = np.where(ctrl_mask_full)[0]
print(f"Using all {len(ctrl_indices_full)} control cells across all perturbations.")

# Process unreliable perturbations in batches of BATCH_SIZE (15)
for batch_idx, i in enumerate(range(0, len(available_perts), BATCH_SIZE)):
    batch_perts = available_perts[i:i + BATCH_SIZE]
    print(f"\nProcessing batch {batch_idx + 1}: {len(batch_perts)} perturbations")

    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    axes = axes.flatten()

    for idx, target_pert in enumerate(batch_perts):
        ax = axes[idx]
        print(f"  -> Computing PS score for: {target_pert}")

        pert_mask = (adata_full.obs["perturbation"] == target_pert).values
        n_pert = pert_mask.sum()

        if n_pert == 0:
            ax.text(0.5, 0.5, f"No cells for {target_pert}", ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{target_pert} [unreliable]", fontsize=9, fontweight='bold', color='red')
            continue

        # Use ALL control cells combined with this perturbation's cells (NO downsampling)
        pert_indices = np.where(pert_mask)[0]
        sub_indices = np.concatenate([pert_indices, ctrl_indices_full])
        adata_sub = adata_full[sub_indices].copy()

        sub_pert_mask = (adata_sub.obs["perturbation"] == target_pert).values
        sub_ctrl_mask = (adata_sub.obs["perturbation"] == CONTROL_LABEL).values

        # Step 1: Identify downstream DEGs using Wilcoxon rank-sum test
        sc.tl.rank_genes_groups(
            adata_sub, 
            groupby="perturbation", 
            groups=[target_pert], 
            reference=CONTROL_LABEL, 
            method="wilcoxon"
        )
        de_df = sc.get.rank_genes_groups_df(adata_sub, group=target_pert)
        
        # Order by p-values and filter by absolute log fold change > 0.1
        de_df = de_df.sort_values(by="pvals", ascending=True)
        filtered_de = de_df[ (de_df["logfoldchanges"].abs() > 0.1)]
        # If fewer than 10 pass, fall back to the top 10 ranked genes
        if len(filtered_de) < 10:
            filtered_de = de_df.head(min(10, len(de_df)))
        print(f"Perturbation {target_pert}: {len(filtered_de)} DEGs found after filtering")
        
        target_genes = filtered_de["names"].head(50).values
        if len(target_genes) < 5:
            target_genes = de_df["names"].head(20).values

        # Step 2: Compute PS Score via Ridge Regression beta weights
        X_data = adata_sub.X.toarray() if hasattr(adata_sub.X, "toarray") else adata_sub.X
        gene_names = adata_sub.var_names

        valid_target_genes = [g for g in target_genes if g in gene_names]
        if len(valid_target_genes) == 0:
            valid_target_genes = gene_names[:20]

        gene_indices = [np.where(gene_names == g)[0][0] for g in valid_target_genes]
        Y = X_data[:, gene_indices]
        D = sub_pert_mask.astype(int).reshape(-1, 1)

        ridge = Ridge(alpha=0.01)
        ridge.fit(D, Y)
        beta_weights = ridge.coef_[:, 0]

        Y_control_mean = Y[sub_ctrl_mask].mean(axis=0)
        Y_centered = Y - Y_control_mean

        ps_scores = np.dot(Y_centered, beta_weights)

        # Min-Max scale PS scores between 0 and 1
        ps_min, ps_max = ps_scores.min(), ps_scores.max()
        ps_scores_normalized = (ps_scores - ps_min) / (ps_max - ps_min + 1e-8)
        adata_sub.obs["ps_score"] = ps_scores_normalized

        ctrl_vals = adata_sub.obs.loc[sub_ctrl_mask, "ps_score"].values
        pert_vals = adata_sub.obs.loc[sub_pert_mask, "ps_score"].values

        # Plot distributions
        sns.kdeplot(
            ctrl_vals,
            ax=ax,
            color="#333333",
            label="Control",
            fill=True,
            alpha=0.15,
            linewidth=1.5
        )

        sns.kdeplot(
            pert_vals,
            ax=ax,
            color="#D55E00",  # Vermillion for unreliable
            label=target_pert,
            fill=True,
            alpha=0.15,
            linewidth=1.5
        )

        ax.set_title(f"{target_pert} | Quality: unreliable\n(n={n_pert})", fontsize=9, fontweight='bold')
        ax.set_xlabel("PS Score", fontsize=7)
        ax.set_ylabel("Cell Density", fontsize=7)
        ax.legend(loc="upper right", fontsize=5, framealpha=0.8)

    # Hide unused subplots in the final batch grid
    for idx in range(len(batch_perts), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(f"PS Score Density: Unreliable Perturbations Batch {batch_idx + 1} vs Control (Full Controls)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = OUT_DIR / f"ps_score_unreliable_batch_{batch_idx + 1}_{DATASET}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved batch plot to: {out_path}")

print("Done processing all unreliable perturbation batches!")