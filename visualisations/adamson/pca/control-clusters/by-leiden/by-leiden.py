import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc

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

# 1. Run PCA & Leiden clustering
sc.pp.pca(adata_ctrl, n_comps=20)
pcs = adata_ctrl.obsm["X_pca"]
var_ratio = adata_ctrl.uns["pca"]["variance_ratio"]

sc.pp.neighbors(adata_ctrl, n_neighbors=30, n_pcs=10)
sc.tl.leiden(adata_ctrl, resolution=0.15, key_added="leiden_ctrl")

print("Leiden cluster counts:\n", adata_ctrl.obs["leiden_ctrl"].value_counts())

# 2. Find DE genes between the Leiden clusters
sc.tl.rank_genes_groups(adata_ctrl, groupby="leiden_ctrl", method="wilcoxon")
df_c0 = sc.get.rank_genes_groups_df(adata_ctrl, group="0")

print("\n=== TOP 15 GENES HIGHER IN LEIDEN 0 ===")
print(df_c0.head(15)[["names", "scores", "logfoldchanges", "pvals_adj"]].to_string(index=False))

print("\n=== TOP 15 GENES HIGHER IN LEIDEN 1 ===")
print(df_c0.tail(15).iloc[::-1][["names", "scores", "logfoldchanges", "pvals_adj"]].to_string(index=False))

# 3. Single plot for Leiden clusters
fig, ax = plt.subplots(figsize=(7, 6))

for cl in sorted(adata_ctrl.obs["leiden_ctrl"].unique()):
    m = (adata_ctrl.obs["leiden_ctrl"] == cl).to_numpy()
    ax.scatter(
        pcs[m, 0], pcs[m, 1],
        s=8, alpha=0.6,
        label=f"Leiden {cl} (n={m.sum()})",
        rasterized=True,
    )

ax.set_title(f"{DATASET} Controls: Leiden Clustering (resolution=0.15)", fontweight="bold")
ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
ax.legend(loc="lower right")

fig.tight_layout()
out_path = OUT_DIR / f"pca12_{DATASET}_controls_leiden.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"\nSaved plot to: {out_path}")


# === TOP 15 GENES HIGHER IN LEIDEN 0 ===
#    names    scores  logfoldchanges  pvals_adj
#      NCL 64.611069        1.740271        0.0
#   HNRNPU 57.665054        1.513361        0.0
#      SET 54.805779        1.150889        0.0
#    HSPD1 50.816097        0.967917        0.0
#   EEF1A1 50.677902        1.608859        0.0
# HSP90AB1 48.990738        0.828910        0.0
# HIST1H1E 46.476208        3.601564        0.0
#     XPO1 45.973186        1.754689        0.0
#    PRKDC 45.891636        1.395094        0.0
#      LBR 44.589405        1.698360        0.0
#     CANX 42.449490        1.104821        0.0
#  HNRNPA3 41.581951        1.136800        0.0
#     PTMA 41.045841        0.612215        0.0
#  SMARCA5 40.602592        1.501583        0.0
#    SETD2 40.450768        1.684761        0.0

# === TOP 15 GENES HIGHER IN LEIDEN 1 ===
#  names     scores  logfoldchanges     pvals_adj
#  RPL21 -46.530655       -0.575707  0.000000e+00
#   EIF1 -46.379143       -0.516171  0.000000e+00
#  RPS23 -42.444962       -0.378204  0.000000e+00
# MT-CO3 -39.742958       -0.385495  0.000000e+00
#  RPL12 -36.389351       -0.335940 1.617499e-287
# RPL35A -34.541645       -0.320637 3.650426e-259
#  RPS3A -33.748512       -0.390847 1.932090e-247
#   RPL7 -33.351215       -0.424516 1.131535e-241
# RPS15A -31.441507       -0.314841 6.855116e-215
# MT-CO2 -31.060081       -0.429746 9.970662e-210
#  ATP5I -30.750397       -0.601644 1.384113e-205
# EIF2S2 -30.555582       -0.546448 5.174906e-203
#   RPS6 -29.631531       -0.331309 6.050359e-191
# MT-CYB -28.928406       -0.324374 5.152692e-182
# MT-ND2 -27.224268       -0.259102 2.851469e-161