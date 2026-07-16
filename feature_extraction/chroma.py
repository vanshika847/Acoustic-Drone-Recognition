"""Chroma feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray

from feature_extraction.mfcc import MFCCExtractionError, _validate_waveform


@dataclass(frozen=True, slots=True)
class ChromaConfig:

    n_chroma: int = 12

    n_fft: int = 2048

    hop_length: int = 512

    win_length: int = 2048

    window: str = "hann"

    center: bool = True

    tuning: float | None = None

    normalize: bool = True

    def __post_init__(self) -> None:
        if self.n_chroma <= 0:
            raise MFCCExtractionError("n_chroma must be greater than zero.")
        if self.n_fft <= 0:
            raise MFCCExtractionError("n_fft must be greater than zero.")
        if self.hop_length <= 0:
            raise MFCCExtractionError("hop_length must be greater than zero.")
        if self.win_length <= 0:
            raise MFCCExtractionError(
                "win_length must be greater than zero."
            )

        if self.win_length > self.n_fft:
            raise MFCCExtractionError(
                "win_length cannot exceed n_fft."
            )
        if self.tuning is not None and not np.isfinite(self.tuning):
            raise MFCCExtractionError(
                "tuning must be a finite value or None."
            )


def extract_chroma(
    waveform: NDArray[np.floating],
    sample_rate: int,
    config: ChromaConfig | None = None,
) -> NDArray[np.float32]:
    """Extract chroma features from a mono waveform.

    Args:
        waveform: One-dimensional mono audio samples.
        sample_rate: Waveform sample rate in Hertz.
        config: Chroma parameters. Defaults to :class:`ChromaConfig`.

    Returns:
        Chroma matrix with shape ``(n_chroma, n_frames)`` as float32.

    Raises:
        MFCCExtractionError: If the waveform or sample rate is invalid.
    """

    settings = config or ChromaConfig()
    audio = _validate_waveform(waveform)
    if sample_rate <= 0:
        raise MFCCExtractionError("sample_rate must be greater than zero.")

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate,
        n_chroma=settings.n_chroma,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        win_length=settings.win_length,
        window=settings.window,
        center=settings.center,
        tuning=settings.tuning,
    )

    if settings.normalize:
        mean = np.mean(chroma)

        std = np.std(chroma)

        chroma = (
            chroma - mean
        ) / (
            std + 1e-8
        )

    return np.asarray(chroma, dtype=np.float32)