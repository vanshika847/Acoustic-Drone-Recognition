import pandas as pd
from pathlib import Path

root = Path("datasets/processed/manifests")

for name in ["train_segments.csv", "validation_segments.csv", "test_segments.csv"]:
    path = root / name

    if path.exists():
        df = pd.read_csv(path)
        print(f"{name}: {len(df)} rows")
        print(df["binary_label"].value_counts().to_dict())
    else:
        print(f"{name}: MISSING")
