# upload_datasets_to_hf.py
# Uploads Phase 1 datasets to a private HuggingFace dataset repo.
# Run once from your local machine before moving to the cluster.
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login   (paste your HF token when prompted)
#
# Run: python scripts/upload_datasets_to_hf.py

import os
import tarfile
from huggingface_hub import HfApi

# ── Config — fill these in ─────────────────────────────────────────────────────
HF_USERNAME  = "SarthakVishnu"          # e.g. "s2880814"
REPO_NAME    = "dissertation-dataset"      # must match what you created on HF
# ──────────────────────────────────────────────────────────────────────────────

REPO_ID      = f"{HF_USERNAME}/{REPO_NAME}"
DATASETS_DIR = r"D:\UoE AI\Dissertation\IPP Draft\datasets"
README_PATH  = r"D:\UoE AI\Dissertation\IPP Draft\datasets\HF_README.md"

api = HfApi()

# ── 0. Upload README as the dataset card ──────────────────────────────────────
print("Uploading README (dataset card)...")
api.upload_file(
    path_or_fileobj=README_PATH,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("  Done: README.md")

# ── 1. Small files — upload directly ──────────────────────────────────────────
SMALL_FILES = [
    "feature_table.parquet",
    "filings_index.csv",
    "volatility_labels.csv",
    "permno_linkage.csv",
    "sp500_union_constituents(1).csv",
]

print("Uploading individual files...")
for fname in SMALL_FILES:
    fpath = os.path.join(DATASETS_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  [SKIP] {fname} not found")
        continue
    print(f"  Uploading {fname}...")
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo=fname,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"  Done: {fname}")


# ── 2. sp500_1A folder — tar.gz first, then upload ────────────────────────────
SP500_FOLDER = os.path.join(DATASETS_DIR, "sp500_1A")
ARCHIVE_PATH = os.path.join(DATASETS_DIR, "sp500_1A.tar.gz")

print("\nCreating sp500_1A.tar.gz (this may take a minute)...")
with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
    tar.add(SP500_FOLDER, arcname="sp500_1A")
size_mb = os.path.getsize(ARCHIVE_PATH) / (1024 * 1024)
print(f"  Archive size: {size_mb:.1f} MB")

print("  Uploading sp500_1A.tar.gz to HuggingFace...")
api.upload_file(
    path_or_fileobj=ARCHIVE_PATH,
    path_in_repo="sp500_1A.tar.gz",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("  Done: sp500_1A.tar.gz")

# Clean up local archive
os.remove(ARCHIVE_PATH)
print("  Local archive removed.")

print(f"\nAll files uploaded to: https://huggingface.co/datasets/{REPO_ID}")
print("Repo is private — share access via HF token only.")
