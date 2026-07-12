"""
scan_datasets.py

Scans every dataset inside datasets/raw
and collects metadata for every audio file.
"""

from pathlib import Path

# Root folder of all raw datasets
RAW_DATASET_DIR = Path("datasets/raw")

# Supported audio extensions
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a"
}


def scan_dataset():

    print("=" * 50)
    print("Scanning datasets...")
    print("=" * 50)

    total_files = 0

    # Loop through each dataset folder
    for dataset in RAW_DATASET_DIR.iterdir():

        if not dataset.is_dir():
            continue

        print(f"\nDataset : {dataset.name}")

        dataset_count = 0

        # Search every subfolder
        for file in dataset.rglob("*"):

            if file.suffix.lower() in AUDIO_EXTENSIONS:

                dataset_count += 1
                total_files += 1

        print(f"Audio Files : {dataset_count}")

    print("\n" + "=" * 50)
    print(f"TOTAL AUDIO FILES : {total_files}")
    print("=" * 50)


if __name__ == "__main__":
    scan_dataset()