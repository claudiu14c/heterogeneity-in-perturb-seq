
import os
import urllib.request
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import pyreadr

print("Start")

DATASET = "adamson16"
PERTURBATIONS = ["STT3A", "OST4", "EIF2S1", "CARS"]
CONTROL_LABEL = "control"

# Adamson16 is human data.
SPECIES = "human"

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))

OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 0. LOAD REFERENCE neuroRef
# ==========================================

print("Loading neuroRef reference PCA...")

REFERENCE_PATH = OUT_DIR / "../../../data/neuroRef.rda"

result = pyreadr.read_r(str(REFERENCE_PATH))

if "neuroRef" not in result:
    raise ValueError(
        f"Could not find neuroRef in file. Available objects: {list(result.keys())}"
    )

neuroRef = result["neuroRef"]

print("Reference shape:", neuroRef.shape)
print("Reference columns:", neuroRef.columns.tolist())

# Extract the original reference rotation matrix.
# The first two columns contain PC1 and PC2 weights.

rotation = neuroRef.iloc[:, :2].copy()
rotation.columns = ["PC1", "PC2"]

# Reference gene symbols are used for human data.
if SPECIES == "human":
    reference_genes = neuroRef["symbol"].astype(str).str.upper()
else:
    reference_genes = neuroRef["ensembl"].astype(str)

rotation.index = reference_genes

# Remove missing and duplicated identifiers.
valid = (
    rotation.index.notna()
    & (rotation.index != "NAN")
    & ~rotation.index.duplicated()
)

rotation = rotation.loc[valid]

print(f"Reference genes available: {len(rotation)}")


# ==========================================
# 1. LOAD DATASET
# ==========================================

print(f"Loading dataset from: {DATA_PATH}")

adata_full = ad.read_h5ad(DATA_PATH)

print(f"Full dataset shape: {adata_full.shape}")


# ==========================================
# 2. PROJECT ONTO neuroRef
# ==========================================

print("Projecting cells into the original tricycle reference space...")

# IMPORTANT:
# The input must already be library-size normalized and log-transformed.
# Do NOT apply normalize_total/log1p again if this is already true.

# Identify the appropriate gene identifiers.
# For human data, prefer gene symbols.

gene_candidates = [
    adata_full.var_names,
]

if SPECIES == "human":
    possible_symbol_columns = [
        "gene_symbols",
        "gene_symbol",
        "symbol",
        "gene_name",
        "feature_name",
    ]

    for col in possible_symbol_columns:
        if col in adata_full.var.columns:
            gene_candidates.append(adata_full.var[col])

# Select the identifier type with the greatest overlap with neuroRef.
reference_set = set(rotation.index)

best_gene_names = None
best_overlap = 0
best_source = None

for i, candidate in enumerate(gene_candidates):

    candidate = pd.Index(
        pd.Series(candidate).astype(str).str.upper()
    )

    overlap = len(set(candidate) & reference_set)

    if overlap > best_overlap:
        best_overlap = overlap
        best_gene_names = candidate
        best_source = (
            "var_names" if i == 0
            else possible_symbol_columns[i - 1]
        )

print(f"Best gene identifier source: {best_source}")
print(f"Overlapping reference genes: {best_overlap}")

if best_gene_names is None or best_overlap < 20:
    raise ValueError(
        "Too few genes match neuroRef. Check gene identifiers "
        "and make sure human gene symbols are available."
    )

# Match genes while preserving reference order.
gene_to_position = {}

for i, gene in enumerate(best_gene_names):
    if gene not in gene_to_position:
        gene_to_position[gene] = i

common_genes = [
    gene for gene in rotation.index
    if gene in gene_to_position
]

if len(common_genes) < 20:
    raise ValueError("Insufficient overlapping reference genes.")

print(f"Using {len(common_genes)} genes for projection.")

# Extract expression for matching genes.
gene_positions = [
    gene_to_position[g] for g in common_genes
]

X_cc = adata_full.X[:, gene_positions]

# The reference is small (~500 genes), so densification is manageable.
if hasattr(X_cc, "toarray"):
    X_cc = X_cc.toarray()
else:
    X_cc = np.asarray(X_cc)

X_cc = X_cc.astype(np.float64)

if not np.isfinite(X_cc).all():
    raise ValueError("Expression matrix contains NaN or infinite values.")

# Extract corresponding reference weights.
W = rotation.loc[common_genes, ["PC1", "PC2"]].to_numpy()

# EXACT tricycle projection:
# Mean-center each gene across cells, without scaling.
X_centered = X_cc - X_cc.mean(axis=0, keepdims=True)

# Project onto the frozen reference PCA.
X_tricycle = X_centered @ W

# Store the embedding.
adata_full.obsm["X_tricycle"] = X_tricycle

# Calculate continuous cell-cycle position.
theta = np.arctan2(
    X_tricycle[:, 1],
    X_tricycle[:, 0]
)

theta = np.mod(theta, 2 * np.pi)

adata_full.obs["tricycle_angle"] = theta

print("Tricycle projection completed.")
print("Embedding shape:", X_tricycle.shape)


# ==========================================
# 3. COMPUTE HVGS AND SUBSET GENES
# ==========================================

print("Computing global highly variable genes...")

sc.pp.highly_variable_genes(
    adata_full,
    n_top_genes=2000,
    flavor="seurat"
)

hvg_mask = adata_full.var["highly_variable"].values

print(f"Number of global HVGs selected: {hvg_mask.sum()}")

adata_full = adata_full[:, hvg_mask].copy()


# ==========================================
# 4. CREATE CONTROL AND PERTURBATION MASKS
# ==========================================

ctrl_mask = (
    adata_full.obs["perturbation"] == CONTROL_LABEL
).values

pert_mask = (
    adata_full.obs["perturbation"].isin(PERTURBATIONS)
).values

n_ctrl = ctrl_mask.sum()
n_pert_total = pert_mask.sum()

print(f"Number of control cells: {n_ctrl}")
print(f"Total perturbation cells: {n_pert_total}")


# ==========================================
# 5. FIT PCA ON CONTROL CELLS ONLY
# ==========================================

adata_ctrl = adata_full[ctrl_mask].copy()

print("Running PCA on control cells only...")

X_ctrl = adata_ctrl.X

if hasattr(X_ctrl, "toarray"):
    X_ctrl = X_ctrl.toarray()
else:
    X_ctrl = np.asarray(X_ctrl)

pca_model = sc.pp.pca if False else None

from sklearn.decomposition import PCA

pca_model = PCA(
    n_components=50,
    random_state=42
)

X_pca_ctrl = pca_model.fit_transform(X_ctrl)

adata_ctrl.obsm["X_pca"] = X_pca_ctrl


# ==========================================
# 6. PROJECT COMBINED CELLS ONTO FROZEN PCA
# ==========================================

combined_mask = ctrl_mask | pert_mask

adata_combined = adata_full[combined_mask].copy()

print("Projecting controls + perturbations onto frozen control PCA...")

X_combined = adata_combined.X

if hasattr(X_combined, "toarray"):
    X_combined = X_combined.toarray()
else:
    X_combined = np.asarray(X_combined)

X_pca_combined = pca_model.transform(X_combined)

adata_combined.obsm["X_pca"] = X_pca_combined


# ==========================================
# 7. RUN t-SNE
# ==========================================

print("Running t-SNE on first 12 frozen PCs...")

sc.tl.tsne(
    adata_combined,
    n_pcs=12,
    random_state=42
)

tsne_coords = adata_combined.obsm["X_tsne"]


# ==========================================
# 8. GENERATE 6-PANEL FIGURE
# ==========================================

print("Generating visualization figure...")

fig, axes = plt.subplots(
    2, 3,
    figsize=(18, 12)
)

axes = axes.flatten()

cmap = plt.cm.twilight

is_ctrl_comb = (
    adata_combined.obs["perturbation"] == CONTROL_LABEL
).values

angles_comb = adata_combined.obs["tricycle_angle"].values


# Panel 1: Controls

ax = axes[0]

ax.scatter(
    tsne_coords[~is_ctrl_comb, 0],
    tsne_coords[~is_ctrl_comb, 1],
    c="lightgray",
    s=4,
    alpha=0.15,
    rasterized=True
)

sc_ctrl = ax.scatter(
    tsne_coords[is_ctrl_comb, 0],
    tsne_coords[is_ctrl_comb, 1],
    c=angles_comb[is_ctrl_comb],
    cmap=cmap,
    vmin=0,
    vmax=2*np.pi,
    s=6,
    alpha=0.7,
    rasterized=True
)

ax.set_title(f"Control Cells (n={n_ctrl})")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")


# Panels 2-5: Individual perturbations

for i, pert in enumerate(PERTURBATIONS):

    ax = axes[i + 1]

    is_pert = (
        adata_combined.obs["perturbation"] == pert
    ).values

    n_pert = is_pert.sum()

    ax.scatter(
        tsne_coords[~is_pert, 0],
        tsne_coords[~is_pert, 1],
        c="lightgray",
        s=4,
        alpha=0.15,
        rasterized=True
    )

    ax.scatter(
        tsne_coords[is_pert, 0],
        tsne_coords[is_pert, 1],
        c=angles_comb[is_pert],
        cmap=cmap,
        vmin=0,
        vmax=2*np.pi,
        s=6,
        alpha=0.8,
        rasterized=True
    )

    ax.set_title(f"Perturbation: {pert} (n={n_pert})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")


# Panel 6: ACTUAL tricycle reference embedding

ax_circle = axes[5]

ax_circle.scatter(
    X_tricycle[:, 0],
    X_tricycle[:, 1],
    c=theta,
    cmap=cmap,
    vmin=0,
    vmax=2*np.pi,
    s=4,
    alpha=0.6,
    rasterized=True
)

ax_circle.set_title(
    "Tricycle Reference Embedding",
    fontweight="bold"
)

ax_circle.set_xlabel("Reference PC1")
ax_circle.set_ylabel("Reference PC2")
ax_circle.set_aspect("equal", adjustable="datalim")


# Shared cyclic colorbar

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])

cbar = fig.colorbar(sc_ctrl, cax=cbar_ax)

cbar.set_ticks([
    0,
    np.pi/2,
    np.pi,
    3*np.pi/2,
    2*np.pi
])

cbar.set_ticklabels([
    "G1 (0)",
    "S (π/2)",
    "G2M (π)",
    "Late G2M (3π/2)",
    "End (2π)"
])

cbar.set_label(
    "Inferred Cell Cycle Angle",
    rotation=270,
    labelpad=15
)

fig.suptitle(
    f"Cell Cycle Progression Analysis | Dataset: {DATASET}",
    fontsize=16,
    y=0.95
)

fig.tight_layout(rect=[0, 0, 0.90, 0.93])

tsne_out_path = OUT_DIR / "tsne_cell_cycle_tricycle_style.png"

fig.savefig(tsne_out_path, dpi=200)

plt.close(fig)

print("Saved:", tsne_out_path)
print("Done!")
