import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")  # No display on cluster; render straight to file
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

print("Start")

DATASET = "adamson16"
PERTURBATIONS = ["STT3A", "OST4", "EIF2S1", "CARS"]
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

# ==========================================
# 1. OFFICIAL SCANPY CELL CYCLE SCORING
# (Using standard Regev lab S-phase and G2M-phase gene lists)
# ==========================================
print("Running official Scanpy cell cycle scoring...")

s_genes = [
    "MCM4", "PCNA", "MSH2", "FEN1", "RFC2", "RRM1", "UNG", "GINS2", "MCM5",
    "HELLS", "RFC4", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", 
    "CCNE2", "UBR7", "POLD3", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", 
    "EXO1", "TIPIN", "DSCC1", "BLM", "USP1", "CLSPN", "POLA1", "CHAF1B", 
    "BRIP1", "E2F8"
]

g2m_genes = [
    "TOP2A", "CDK1", "CCNB1", "UBE2C", "AURKA", "AURKB", "MKI67", "HMGB2", 
    "CDCA2", "CDCA3", "CDCA8", "CKS1B", "CKS2", "CTPS1", "DCK", "DTL", "DUT", 
    "E2F1", "ECT2", "EZH2", "GAS2L3", "GINS1", "H2AFX", "HIST1H4C", "HJURP", 
    "KIF11", "KIF20B", "KIF23", "KIF2C", "LBR", "MKI67", "NCAPD2", "NUCKS1", 
    "NUSAP1", "PCLAF", "PLK1", "PRC1", "PTP4A2", "RAD54L", "RMI1", "ROCC1"
]

# Filter lists to genes actually present in the dataset
s_genes_avail = [g for g in s_genes if g in adata_full.var_names]
g2m_genes_avail = [g for g in g2m_genes if g in adata_full.var_names]

# Run Scanpy's cell cycle scoring function
sc.tl.score_genes_cell_cycle(adata_full, s_genes=s_genes_avail, g2m_genes=g2m_genes_avail)
print("Cell cycle scores and phases successfully calculated!")
# Note: This adds adata.obs['S_score'], adata.obs['G2M_score'], and adata.obs['phase']

# ==========================================
# 2. COMPUTE HVGS AND SUBSET FOR FROZEN PCA
# ==========================================
print("Computing Global Highly Variable Genes (HVGs) on all cells...")
sc.pp.highly_variable_genes(adata_full, n_top_genes=2000, flavor="seurat")
hvg_mask = adata_full.var["highly_variable"].values
print(f"Number of global HVGs selected: {hvg_mask.sum()}")

# Subset dataset to HVGs (cell cycle scores in .obs are safely preserved)
adata_full = adata_full[:, hvg_mask].copy()

# Create masks for controls and specified perturbations
ctrl_mask = (adata_full.obs["perturbation"] == CONTROL_LABEL).values
pert_mask = adata_full.obs["perturbation"].isin(PERTURBATIONS).values

n_ctrl = ctrl_mask.sum()
n_pert_total = pert_mask.sum()
print(f"Number of control cells available: {n_ctrl}")
print(f"Total perturbation cells across {PERTURBATIONS}: {n_pert_total}")

# ==========================================
# 3. FIT PCA ON CONTROL CELLS ONLY
# ==========================================
adata_ctrl = adata_full[ctrl_mask].copy()
print("Running PCA on control cells only (n_comps=50)...")

X_ctrl = adata_ctrl.X.toarray() if hasattr(adata_ctrl.X, "toarray") else adata_ctrl.X
pca_model = PCA(n_components=50, random_state=42)
X_pca_ctrl = pca_model.fit_transform(X_ctrl)
adata_ctrl.obsm["X_pca"] = X_pca_ctrl

# ==========================================
# 4. PROJECT COMBINED SUBSET ONTO FROZEN CONTROL PCA
# ==========================================
combined_mask = ctrl_mask | pert_mask
adata_combined = adata_full[combined_mask].copy()

print("Projecting combined controls + perturbations onto frozen control PCA...")
X_combined = adata_combined.X.toarray() if hasattr(adata_combined.X, "toarray") else adata_combined.X
X_pca_combined = pca_model.transform(X_combined)

adata_combined.obsm["X_pca"] = X_pca_combined

# ==========================================
# 5. RUN T-SNE ON FIRST 12 FROZEN PCS
# ==========================================
print("Running t-SNE on the first 12 frozen PCs...")
sc.tl.tsne(adata_combined, n_pcs=12, random_state=42)
tsne_coords = adata_combined.obsm["X_tsne"]

# ==========================================
# 6. PLOT 6-PANEL FIGURE (Controls + 4 Perts + Phase Distribution QC)
# ==========================================
print("Generating 6-panel visualization figure...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Using the twilight colormap for the continuous G2M score gradient
cmap = plt.cm.twilight
is_ctrl_comb = (adata_combined.obs["perturbation"] == CONTROL_LABEL).values
# We can visualize either 'G2M_score' or 'S_score' (using G2M score here as an example)
scores_comb = adata_combined.obs["G2M_score"].values
vmin, vmax = scores_comb.min(), scores_comb.max()

# Panel 1: Control Cells Only
ax = axes[0]
ax.scatter(tsne_coords[~is_ctrl_comb, 0], tsne_coords[~is_ctrl_comb, 1], c="lightgray", s=4, alpha=0.15, rasterized=True)
sc_ctrl = ax.scatter(
    tsne_coords[is_ctrl_comb, 0], tsne_coords[is_ctrl_comb, 1], 
    c=scores_comb[is_ctrl_comb], cmap=cmap, vmin=vmin, vmax=vmax, 
    s=6, alpha=0.7, rasterized=True
)
ax.set_title(f"Control Cells (n={n_ctrl})")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")

# Panels 2-5: Each Perturbation individually
for i, pert in enumerate(PERTURBATIONS):
    ax = axes[i + 1]
    is_pert = (adata_combined.obs["perturbation"] == pert).values
    n_pert = is_pert.sum()
    
    # Background gray for controls and other cells
    ax.scatter(tsne_coords[~is_pert, 0], tsne_coords[~is_pert, 1], c="lightgray", s=4, alpha=0.15, rasterized=True)
    # Foreground colored perturbation cells
    ax.scatter(
        tsne_coords[is_pert, 0], tsne_coords[is_pert, 1], 
        c=scores_comb[is_pert], cmap=cmap, vmin=vmin, vmax=vmax, 
        s=6, alpha=0.8, rasterized=True
    )
    ax.set_title(f"Perturbation: {pert} (n={n_pert})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

# Panel 6: Cell Cycle Phase Distribution Barplot (QC)
ax_qc = axes[5]
phase_counts = adata_full.obs["phase"].value_counts()[['G1', 'S', 'G2M']]
phase_counts.plot(kind='bar', ax=ax_qc, color=['skyblue', 'salmon', 'mediumpurple'], edgecolor='black', alpha=0.8)
ax_qc.set_title("Cell Cycle Phase Counts (QC)", fontweight='bold')
ax_qc.set_xlabel("Phase")
ax_qc.set_ylabel("Cell Count")
ax_qc.tick_params(axis='x', rotation=0)

# Shared colorbar for scores matching twilight range
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(sc_ctrl, cax=cbar_ax)
cbar.set_label("G2M Score (Scanpy official lists / twilight palette)", rotation=270, labelpad=15)

fig.suptitle(f"Scanpy Cell Cycle Scoring Analysis | Dataset: {DATASET}", fontsize=16, y=0.95)
fig.tight_layout(rect=[0, 0, 0.90, 0.93])

tsne_out_path = OUT_DIR / "tsne_scanpy_cell_cycle.png"
fig.savefig(tsne_out_path, dpi=200)
plt.close(fig)
print("Saved cell cycle t-SNE plot:", tsne_out_path)

print("Done!")