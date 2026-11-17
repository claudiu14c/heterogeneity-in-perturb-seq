import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
import pandas as pd

DATASET = "adamson16"
CONTROL_LABEL = "control"

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    f"/cluster/work/boeva/virtual_cell_reasoning/data/{DATASET}/{DATASET}_processed.h5ad",
))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
OUT_DIR.mkdir(parents=True, exist_ok=True)

adata_full = ad.read_h5ad(DATA_PATH)
adata_ctrl = adata_full[adata_full.obs["perturbation"] == CONTROL_LABEL].copy()

# 1. Inspect ALL .obs columns (including any skipped earlier)
print("=== ALL .obs COLUMNS ===")
for col in adata_ctrl.obs.columns:
    print(f"  {col}: {adata_ctrl.obs[col].nunique()} unique | sample: {adata_ctrl.obs[col].iloc[:3].tolist()}")

# 2. Run PCA and split the two visual clusters in PC1-PC2 space
sc.pp.pca(adata_ctrl, n_comps=20)
pcs = adata_ctrl.obsm["X_pca"]

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
adata_ctrl.obs["ctrl_cluster"] = kmeans.fit_predict(pcs[:, :2]).astype(str)
print("\nCluster sizes:\n", adata_ctrl.obs["ctrl_cluster"].value_counts())

# 3. Find differentially expressed genes between Cluster 0 and Cluster 1
sc.tl.rank_genes_groups(adata_ctrl, groupby="ctrl_cluster", method="wilcoxon")
df_markers = sc.get.rank_genes_groups_df(adata_ctrl, group="0")

print("\n=== TOP 15 GENES HIGHER IN CLUSTER 0 ===")
print(df_markers.head(15)[["names", "scores", "logfoldchanges", "pvals_adj"]].to_string(index=False))

print("\n=== TOP 15 GENES HIGHER IN CLUSTER 1 ===")
print(df_markers.tail(15).iloc[::-1][["names", "scores", "logfoldchanges", "pvals_adj"]].to_string(index=False))

# 4. Also check total counts / detected genes calculated directly from .X
adata_ctrl.obs["calc_total_counts"] = np.asarray(adata_ctrl.X.sum(axis=1)).ravel()
adata_ctrl.obs["calc_n_genes"] = np.asarray((adata_ctrl.X > 0).sum(axis=1)).ravel()

# 5. Plot cluster assignment + calculated QC metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for cl, color in [("0", "#e41a1c"), ("1", "#377eb8")]:
    m = (adata_ctrl.obs["ctrl_cluster"] == cl).to_numpy()
    axes[0].scatter(pcs[m, 0], pcs[m, 1], c=color, s=8, alpha=0.6, label=f"Cluster {cl} (n={m.sum()})", rasterized=True)
axes[0].set_title("KMeans (k=2) on PC1-PC2")
axes[0].legend()

for idx, metric in enumerate(["calc_total_counts", "calc_n_genes"], start=1):
    sc_p = axes[idx].scatter(pcs[:, 0], pcs[:, 1], c=adata_ctrl.obs[metric], cmap="viridis", s=8, alpha=0.6, rasterized=True)
    plt.colorbar(sc_p, ax=axes[idx], label=metric)
    axes[idx].set_title(metric)

for ax in axes:
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

fig.tight_layout()
out_path = OUT_DIR / f"pca12_{DATASET}_controls_de_check.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"\nSaved cluster DE diagnostic to: {out_path}")

print(adata_ctrl.obs.groupby("ctrl_cluster")[["ncounts", "ngenes", "percent_mito", "percent_ribo"]].mean())
# Check if the 3 control guides or barcode suffixes map to Cluster 0 vs Cluster 1
print(pd.crosstab(adata_ctrl.obs["ctrl_cluster"], adata_ctrl.obs["guide"]))

# Check cell barcode suffixes (e.g., -1, -2, -3 from 10x CellRanger aggr)
suffixes = adata_ctrl.obs_names.str.split("-").str[-1]
print(pd.crosstab(adata_ctrl.obs["ctrl_cluster"], suffixes))


# === TOP 15 GENES HIGHER IN CLUSTER 0 ===
#  names    scores  logfoldchanges     pvals_adj
#  RPL21 42.987778        0.523242  0.000000e+00
# MT-CO3 42.624432        0.424098  0.000000e+00
#  RPS23 39.579872        0.346722  0.000000e+00
# MT-CYB 37.585369        0.430783 1.618351e-306
#  RPL12 36.002735        0.331110 2.842475e-281
#  RPS3A 34.139015        0.386419 4.722665e-253
#   EIF1 33.577496        0.354588 7.760867e-245
# MT-CO2 33.313385        0.460722 5.097244e-241
#   RPL7 33.244312        0.411471 4.963694e-240
# RPL35A 32.807644        0.297472 8.789280e-234
#   RPS6 32.070919        0.371097 1.814903e-223
# MT-ND2 31.628017        0.316913 2.241177e-217
# RPS15A 30.503246        0.298884 3.140362e-202
#  ATP5I 28.021429        0.529128 8.854507e-171
# MT-ND4 27.793781        0.241626 4.947490e-168

# === TOP 15 GENES HIGHER IN CLUSTER 1 ===
#    names     scores  logfoldchanges  pvals_adj
#      NCL -57.090931       -1.463213        0.0
#   HNRNPU -51.796928       -1.296772        0.0
#      SET -50.142437       -1.008441        0.0
#      LBR -46.595356       -1.669350        0.0
# HIST1H1E -45.255707       -3.021141        0.0
#    HSPD1 -44.836731       -0.817495        0.0
#     PTMA -43.057392       -0.591262        0.0
#     XPO1 -43.048599       -1.518265        0.0
#     TMPO -42.921322       -1.484599        0.0
# HSP90AB1 -42.021255       -0.695012        0.0
#   HNRNPD -39.750999       -0.819399        0.0
#   EEF1A1 -39.288353       -1.160449        0.0
#    PRKDC -39.049595       -1.114883        0.0
#  HNRNPA3 -39.011215       -1.020174        0.0
# HIST1H1D -38.981792       -2.799313        0.0


# Cluster 0 (n = 3,227): Almost exclusively mitochondrial (MT-CO3, MT-CYB, MT-CO2, MT-ND2, MT-ND4, ATP5I) and cytoplasmic ribosomal (RPL21, RPS23, RPL12, RPS3A, RPL7, RPS6) transcripts.

# Cluster 1 (n = 3,989): Almost exclusively nuclear / chromatin / S-phase replication proteins with massive negative log-fold changes (HIST1H1E at -3.02, HIST1H1D at -2.80, LBR [Lamin B Receptor], TMPO [Lamina-associated polypeptide 2], NCL [Nucleolin], HNRNPU, HNRNPD, HNRNPA3, XPO1, PRKDC).