"""
Audio processing utilities.
"""

import numpy as np
import librosa


def normalize_audio(audio, target_peak=0.99):
    """
    Peak normalization.
    """

    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    return audio * (target_peak / peak)


def remove_dc_offset(audio):
    """
    Remove DC offset.
    """

    return audio - np.mean(audio)


def trim_silence(audio, top_db=25):
    """
    Remove leading and trailing silence.
    """

    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db
    )

    return trimmed


def calculate_peak(audio):
    return float(np.max(np.abs(audio)))


def calculate_rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))