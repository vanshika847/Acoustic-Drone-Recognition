from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CHANGE THIS ONLY IF YOUR PATH IS DIFFERENT
# ============================================================

PROJECT = Path(r"D:\Acoustic-Drone-Recognition")

TRAIN_MANIFEST = PROJECT / "manifests" / "train_manifest.csv"

# Put the hard-example indices from the latest run here.
HARD_INDICES = [
    2837,
    10984,
    5305,
    12082,
    2821,
    8238,
    14369,
    8956,
    9959,
    12762,
]


def main():

    print("=" * 80)
    print("HARD EXAMPLE DIAGNOSTIC")
    print("=" * 80)

    if not TRAIN_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{TRAIN_MANIFEST}"
        )

    df = pd.read_csv(TRAIN_MANIFEST)

    print(f"\nManifest: {TRAIN_MANIFEST}")
    print(f"Total samples: {len(df)}")

    # --------------------------------------------------------
    # Detect index column
    # --------------------------------------------------------

    possible_index_columns = [
        "sample_index",
        "index",
        "idx",
    ]

    index_col = None

    for col in possible_index_columns:
        if col in df.columns:
            index_col = col
            break

    if index_col is None:
        print("\nAvailable columns:")
        print(df.columns.tolist())

        raise ValueError(
            "Could not find sample index column."
        )

    df[index_col] = pd.to_numeric(
        df[index_col],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Extract hard examples
    # --------------------------------------------------------

    hard_df = df[
        df[index_col].isin(HARD_INDICES)
    ].copy()

    if hard_df.empty:
        print("\nNO HARD EXAMPLES FOUND.")
        print("This probably means the manifest index column")
        print("uses a different indexing scheme.")

        print("\nFirst rows:")
        print(df.head())

        return

    hard_df = hard_df.sort_values(
        index_col
    )

    print("\n" + "=" * 80)
    print("HARD EXAMPLES")
    print("=" * 80)

    # Show every column because we want to understand
    # where these samples came from.

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        240,
    )

    print(hard_df.to_string(index=False))

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    if "binary_label" in hard_df.columns:

        print("\n" + "=" * 80)
        print("LABEL DISTRIBUTION")
        print("=" * 80)

        print(
            hard_df["binary_label"]
            .value_counts()
            .sort_index()
        )

    # --------------------------------------------------------
    # Source distribution
    # --------------------------------------------------------

    source_candidates = [
        "source",
        "dataset",
        "dataset_name",
        "source_dataset",
    ]

    for col in source_candidates:

        if col in hard_df.columns:

            print("\n" + "=" * 80)
            print(f"DISTRIBUTION BY {col}")
            print("=" * 80)

            print(
                hard_df[col]
                .value_counts()
            )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    output = PROJECT / "hard_examples_diagnostic.csv"

    hard_df.to_csv(
        output,
        index=False,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"\nDiagnostic saved to:\n{output}"
    )


if __name__ == "__main__":
    main()