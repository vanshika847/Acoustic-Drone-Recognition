"""
validate_audio.py

Validates audio files using the generated metadata.

Checks:
- Corrupted files
- Missing metadata
- Too short
- Too long

Outputs:
- outputs/validation_report.csv
- Validation summary in terminal

Author: Vanshika's Acoustic Drone Recognition Project
"""

from pathlib import Path

import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

METADATA_FILE = Path("datasets/metadata/master_metadata.csv")
OUTPUT_DIR = Path("outputs")

MIN_DURATION = 1.0      # seconds
MAX_DURATION = 30.0     # seconds


# ==========================================================
# Validation Function
# ==========================================================

def validate_audio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate every audio file using metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Master metadata DataFrame.

    Returns
    -------
    pd.DataFrame
        Validation report.
    """

    report = []

    for _, row in df.iterrows():

        issues = []

        # Corrupted file
        if row["status"] != "OK":
            issues.append("Corrupted Audio")

        # Missing duration
        if pd.isna(row["duration_sec"]):
            issues.append("Missing Duration")

        # Too short
        elif row["duration_sec"] < MIN_DURATION:
            issues.append("Too Short")

        # Too long
        elif row["duration_sec"] > MAX_DURATION:
            issues.append("Too Long")

        report.append({

            "dataset": row["dataset"],
            "filename": row["filename"],
            "duration_sec": row["duration_sec"],
            "sample_rate": row["sample_rate"],
            "status": row["status"],
            "issues": ", ".join(issues) if issues else "PASS"

        })

    return pd.DataFrame(report)


# ==========================================================
# Print Summary
# ==========================================================

def print_summary(df, validation_report):

    total = len(validation_report)

    passed = (validation_report["issues"] == "PASS").sum()
    short = (validation_report["issues"] == "Too Short").sum()
    long = (validation_report["issues"] == "Too Long").sum()
    corrupt = validation_report["issues"].str.contains(
        "Corrupted",
        na=False
    ).sum()

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Total Files      : {total}")
    print(f"Passed           : {passed} ({passed/total*100:.2f}%)")
    print(f"Too Short        : {short} ({short/total*100:.2f}%)")
    print(f"Too Long         : {long} ({long/total*100:.2f}%)")
    print(f"Corrupted        : {corrupt} ({corrupt/total*100:.2f}%)")

    print("\n" + "=" * 65)
    print("FILES PER DATASET")
    print("=" * 65)

    print(df["dataset"].value_counts())

    print("\n" + "=" * 65)
    print("AVERAGE DURATION PER DATASET")
    print("=" * 65)

    print(
        df.groupby("dataset")["duration_sec"]
          .mean()
          .round(2)
    )

    print("\n" + "=" * 65)
    print("VALIDATION BY DATASET")
    print("=" * 65)

    dataset_summary = (
        validation_report
        .groupby("dataset")["issues"]
        .apply(lambda x: (x == "PASS").mean() * 100)
        .round(2)
    )

    for dataset, percentage in dataset_summary.items():

        print(f"{dataset:<20} {percentage:>6.2f}% Valid")


# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 65)
    print("Audio Validation Engine")
    print("=" * 65)

    df = pd.read_csv(METADATA_FILE)

    validation_report = validate_audio(df)

    OUTPUT_DIR.mkdir(exist_ok=True)

    output_file = OUTPUT_DIR / "validation_report.csv"

    validation_report.to_csv(output_file, index=False)

    print_summary(df, validation_report)

    print("\nValidation report saved to:")
    print(output_file)

    print("=" * 65)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()