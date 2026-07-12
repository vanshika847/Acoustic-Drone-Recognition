"""
Audio loading utilities.
"""

from pathlib import Path

import librosa
import soundfile as sf


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac"
}


def is_supported_audio(file_path: Path) -> bool:
    """
    Check if the file extension is supported.
    """
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def load_audio(file_path: Path, target_sr: int):
    """
    Load an audio file and convert it to mono.

    Returns
    -------
    audio : np.ndarray
    sample_rate : int
    metadata : dict
    """

    info = sf.info(file_path)

    audio, sr = librosa.load(
        file_path,
        sr=target_sr,
        mono=True
    )

    metadata = {
        "original_samplerate": info.samplerate,
        "original_channels": info.channels,
        "duration": len(audio) / sr,
        "format": info.format,
        "subtype": info.subtype
    }

    return audio, sr, metadata