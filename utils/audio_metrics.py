"""
Audio quality metrics.
"""



import numpy as np


def compute_metrics(audio, sample_rate):

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")

    if len(audio) == 0:
        raise ValueError("audio array is empty")

    duration = len(audio) / sample_rate

    rms = float(np.sqrt(np.mean(audio ** 2)))

    peak = float(np.max(np.abs(audio)))

    dynamic_range = float(20 * np.log10((peak + 1e-8) / (rms + 1e-8)))

    silence_ratio = float(
        np.mean(np.abs(audio) < 0.001)
    )

    return {
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "dynamic_range": dynamic_range,
        "silence_ratio": silence_ratio
    }

    