import os
from pathlib import Path
import anndata as ad

print("Start")

DATA_PATH = Path(os.environ.get(
    "ADAMSON_PATH",
    "/cluster/work/boeva/virtual_cell_reasoning/data/adamson16/adamson16_processed.h5ad",
))

print(f"Loading dataset from: {DATA_PATH}")
adata = ad.read_h5ad(DATA_PATH)

print("\n--- AnnData Summary ---")
print(adata.X)
print(adata.obs)

print(adata.obs.columns.tolist())