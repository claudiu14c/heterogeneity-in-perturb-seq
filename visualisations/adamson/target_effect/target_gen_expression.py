import os
from pathlib import Path
import anndata as ad
import matplotlib
matplotlib.use("Agg")  # no display on the cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

print("Start")

DATASET = "adamson16"
CONTROL_LABEL = "control"  # Change to "control" or "non-targeting" based on dataset
BATCH_SIZE = 15  # 15 perturbations per image (3 rows x 5 columns)

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_CSV_URL = "reliability_adamson.csv"

print(f"Loading dataset from: {DATA_PATH}")
adata_full = ad.read_h5ad(DATA_PATH)
print(f"Full dataset shape: {adata_full.shape}")

# Load the split metadata mapping perturbations to quality labels
print(f"Loading split CSV from: {SPLIT_CSV_URL}")
try:
    split_df = pd.read_csv(SPLIT_CSV_URL)
    # Ensure expected columns exist
    if "perturbation" in split_df.columns and "quality_label" in split_df.columns:
        pert_to_label = dict(zip(split_df["perturbation"].astype(str), split_df["quality_label"].astype(str)))
    else:
        print("Warning: 'perturbation' or 'quality_label' column not found in CSV. Defaulting labels to 'Unknown'.")
        pert_to_label = {}
except Exception as e:
    print(f"Could not load split CSV directly ({e}). Proceeding without quality labels.")
    pert_to_label = {}

# Get all unique perturbations from the dataset (excluding control)
all_perts = adata_full.obs["perturbation"].astype(str).unique()
perts = [p for p in all_perts if p != CONTROL_LABEL]
print(f"Total unique perturbations found: {len(perts)}")

ctrl_mask_full = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
ctrl_indices_full = np.where(ctrl_mask_full)[0]

# Batch perturbations into chunks of 15
for batch_idx, i in enumerate(range(0, len(perts), BATCH_SIZE)):
    batch_perts = perts[i:i + BATCH_SIZE]
    print(f"\nProcessing batch {batch_idx + 1}: {len(batch_perts)} perturbations")

    # Create a 3x5 grid for the 15 subplots
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    axes = axes.flatten()

    for idx, pert in enumerate(batch_perts):
        ax = axes[idx]
        q_label = pert_to_label.get(pert, "N/A")
        
        # Check if gene exists in dataset
        gene_exists = pert in adata_full.var_names
        
        if not gene_exists:
            ax.text(0.5, 0.5, f"Gene {pert}\nnot in dataset", ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{pert} [{q_label}]", fontsize=9, fontweight='bold', color='red')
            continue

        pert_mask = (adata_full.obs["perturbation"] == pert).values
        n_pert = pert_mask.sum()
        n_ctrl = ctrl_mask_full.sum()

        if n_pert == 0:
            ax.text(0.5, 0.5, f"No cells for {pert}", ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{pert} [{q_label}]", fontsize=9, fontweight='bold')
            continue

        pert_indices = np.where(pert_mask)[0]

        # Downsample control cells to match perturbation size
        np.random.seed(42)
        if n_ctrl > n_pert:
            sampled_ctrl_indices = np.random.choice(ctrl_indices_full, size=n_pert, replace=False)
        else:
            sampled_ctrl_indices = ctrl_indices_full

        # Subset data for this specific gene and cell selection
        combined_indices = np.concatenate([pert_indices, sampled_ctrl_indices])
        adata_sub = adata_full[combined_indices, pert].copy()

        expr = adata_sub.X
        if hasattr(expr, "toarray"):
            expr = expr.toarray().flatten()
        else:
            expr = expr.flatten()

        ctrl_vals = expr[len(pert_indices):]
        pert_vals = expr[:len(pert_indices)]

        # Plot Control density or constant spike
        if np.var(ctrl_vals) == 0:
            ax.axvline(ctrl_vals[0], color="#333333", linestyle="--", linewidth=1.5, label=f"Control (Const={ctrl_vals[0]:.2f})")
        else:
            sns.kdeplot(ctrl_vals, ax=ax, color="#333333", label="Control", fill=True, alpha=0.15, linewidth=1.5)

        # Plot Perturbation density or constant spike
        if np.var(pert_vals) == 0:
            ax.axvline(pert_vals[0], color="#009E73", linestyle="--", linewidth=1.5, label=f"{pert} (Const={pert_vals[0]:.2f})")
        else:
            sns.kdeplot(pert_vals, ax=ax, color="#009E73", label=pert, fill=True, alpha=0.15, linewidth=1.5)

        ax.set_title(f"{pert} | Quality: {q_label}\n(n={n_pert})", fontsize=9, fontweight='bold')
        ax.set_xlabel("Expression Value", fontsize=7)
        ax.set_ylabel("Cell Density", fontsize=7)
        ax.legend(loc="upper right", fontsize=5, framealpha=0.8)

    # Hide any unused subplots in the final batch grid
    for idx in range(len(batch_perts), len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(f"Gene Expression Density: Perturbations Batch {batch_idx + 1} vs. Control", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = OUT_DIR / f"perturbations_density_batch_{batch_idx + 1}_{DATASET}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved batch plot to: {out_path}")

print("Done processing all perturbation batches!")