import pandas as pd

files = [
    "outputs/features/train_feature_manifest.csv",
    "outputs/features/validation_feature_manifest.csv",
    "outputs/features/test_feature_manifest.csv",
]

for f in files:
    print("\n" + "=" * 60)
    print(f)
    print("=" * 60)

    df = pd.read_csv(f)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    if "status" in df.columns:
        print("\nSTATUS:")
        print(df["status"].value_counts(dropna=False))

    if "binary_label" in df.columns and "status" in df.columns:
        print("\nLABEL x STATUS:")
        print(pd.crosstab(df["binary_label"], df["status"]))
