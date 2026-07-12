"""
generate_metadata.py

Scans all datasets inside datasets/raw,
extracts metadata from every audio file,
and generates a master_metadata.csv file.

Author: Vanshika's Acoustic Drone Recognition Project
"""

from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ==========================================================
# Configuration
# ==========================================================

RAW_DATASET_DIR = Path("datasets/raw")
OUTPUT_DIR = Path("datasets/metadata")

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
}


# ==========================================================
# Helper Function
# ==========================================================

def get_audio_metadata(file_path: Path) -> dict:
    """
    Reads metadata from an audio file.

    Parameters
    ----------
    file_path : Path
        Path to an audio file.

    Returns
    -------
    dict
        Dictionary containing metadata.
    """

    try:

        info = sf.info(file_path)

        duration = round(info.frames / info.samplerate, 2)

        return {
            "duration_sec": duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "status": "OK",
        }

    except Exception:

        return {
            "duration_sec": None,
            "sample_rate": None,
            "channels": None,
            "subtype": None,
            "status": "CORRUPTED",
        }


# ==========================================================
# Main Scanner
# ==========================================================

def collect_audio_files():
    """
    Scans every dataset and extracts metadata
    for all supported audio files.
    """

    audio_files = []

    datasets = [d for d in RAW_DATASET_DIR.iterdir() if d.is_dir()]

    for dataset in datasets:

        print(f"\nScanning {dataset.name}...")

        files = list(dataset.rglob("*"))

        for file in tqdm(files, desc=dataset.name):

            if file.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                continue

            metadata = get_audio_metadata(file)

            audio_files.append(
                {
                    "dataset": dataset.name,
                    "filename": file.name,
                    "path": str(file),
                    "extension": file.suffix.lower(),

                    "duration_sec": metadata["duration_sec"],
                    "sample_rate": metadata["sample_rate"],
                    "channels": metadata["channels"],
                    "subtype": metadata["subtype"],

                    "file_size_mb": round(
                        file.stat().st_size / (1024 * 1024),
                        3,
                    ),

                    "status": metadata["status"],
                }
            )

    return audio_files


# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 60)
    print("Generating Master Metadata")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = collect_audio_files()

    df = pd.DataFrame(audio_files)

    output_file = OUTPUT_DIR / "master_metadata.csv"

    df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("Metadata generation completed.")
    print(f"Total audio files : {len(df)}")
    print(f"Saved to          : {output_file}")
    print("=" * 60)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()