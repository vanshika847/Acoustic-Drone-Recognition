from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_loader import create_dataloader

loader = create_dataloader(
    PROJECT_ROOT / "outputs/features/train_shard_manifest.csv",
    batch_size=8,
)

batch = next(iter(loader))

print("Batch Loaded\n")

for key, value in batch.items():
    if hasattr(value, "shape"):
        print(f"{key:10} -> {value.shape}")
    else:
        print(f"{key:10} -> {value}")