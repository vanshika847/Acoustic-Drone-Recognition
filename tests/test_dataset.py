from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.feature_dataset import FeatureDataset

dataset = FeatureDataset(
    PROJECT_ROOT / "outputs" / "features" / "train_feature_manifest.csv"
)

print(f"Dataset size: {len(dataset)}")

sample = dataset[0]

print("\nLoaded sample:\n")

for key, value in sample.items():
    if hasattr(value, "shape"):
        print(f"{key:10} -> {value.shape}")
    else:
        print(f"{key:10} -> {value}")