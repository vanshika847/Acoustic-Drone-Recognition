from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_loader import create_dataloader

manifest = (
    PROJECT_ROOT
    / "outputs"
    / "features"
    / "train_shard_manifest.csv"
)

loader = create_dataloader(
    manifest_path=manifest,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

batch = next(iter(loader))

print("BATCH LOADED")
print("Number of batches:", len(loader))
print()

for key, value in batch.items():
    if hasattr(value, "shape"):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {value}")

print()
print("SHARD DATALOADER TEST PASSED")