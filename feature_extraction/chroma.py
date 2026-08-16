"""Chroma feature extraction.

This module extracts chroma features from a mono audio waveform.

Chroma features represent the distribution of spectral energy across
pitch-class bins. They are useful for capturing harmonic and tonal
characteristics of an audio signal.

The returned feature matrix has shape:

    (n_chroma, n_frames)
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray

from feature_extraction.mfcc import (
    MFCCExtractionError,
    _validate_waveform,
)


class ChromaExtractionError(MFCCExtractionError):
    """Raised when chroma feature extraction fails."""


@dataclass(frozen=True, slots=True)
class ChromaConfig:
    """Configuration for chroma feature extraction.

    Attributes:
        n_chroma: Number of chroma bins.
        n_fft: FFT window size in samples.
        hop_length: Number of samples between successive frames.
        win_length: Length of the analysis window in samples.
        window: Window function used during STFT computation.
        center: Whether frames are centered by padding the waveform.
        tuning: Tuning adjustment in fractions of a semitone.
            ``None`` lets librosa estimate tuning automatically.
        normalize: Whether to standardize the resulting feature matrix.
    """

    n_chroma: int = 12
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    window: str = "hann"
    center: bool = True
    tuning: float | None = None
    normalize: bool = True

    def __post_init__(self) -> None:
        """Validate chroma extraction settings."""

        if self.n_chroma <= 0:
            raise ChromaExtractionError(
                "n_chroma must be greater than zero."
            )

        if self.n_fft <= 0:
            raise ChromaExtractionError(
                "n_fft must be greater than zero."
            )

        if self.hop_length <= 0:
            raise ChromaExtractionError(
                "hop_length must be greater than zero."
            )

        if self.win_length <= 0:
            raise ChromaExtractionError(
                "win_length must be greater than zero."
            )

        if self.win_length > self.n_fft:
            raise ChromaExtractionError(
                "win_length cannot exceed n_fft."
            )

        if self.tuning is not None and not np.isfinite(self.tuning):
            raise ChromaExtractionError(
                "tuning must be a finite value or None."
            )


def extract_chroma(
    waveform: NDArray[np.floating],
    sample_rate: int,
    config: ChromaConfig | None = None,
) -> NDArray[np.float32]:
    """Extract chroma features from a mono waveform.

    Args:
        waveform:
            One-dimensional mono audio samples.

        sample_rate:
            Waveform sample rate in Hertz.

        config:
            Chroma extraction parameters. If omitted, the default
            :class:`ChromaConfig` is used.

    Returns:
        Chroma feature matrix with shape
        ``(n_chroma, n_frames)`` as float32.

    Raises:
        ChromaExtractionError:
            If the waveform, sample rate, configuration, or extracted
            features are invalid.
    """

    settings = config or ChromaConfig()

    # ---------------------------------------------------------
    # Validate waveform
    # ---------------------------------------------------------
    try:
        audio = _validate_waveform(waveform)
    except MFCCExtractionError as exc:
        raise ChromaExtractionError(str(exc)) from exc

    # ---------------------------------------------------------
    # Validate sample rate
    # ---------------------------------------------------------
    if sample_rate <= 0:
        raise ChromaExtractionError(
            "sample_rate must be greater than zero."
        )

    # ---------------------------------------------------------
    # Validate waveform length
    # ---------------------------------------------------------
    if audio.size < settings.win_length and not settings.center:
        raise ChromaExtractionError(
            "Waveform is shorter than win_length while center=False."
        )

    # ---------------------------------------------------------
    # Extract chroma features
    # ---------------------------------------------------------
    try:
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

    except Exception as exc:
        raise ChromaExtractionError(
            f"Failed to compute chroma features: {exc}"
        ) from exc

    # ---------------------------------------------------------
    # Convert to float32
    # ---------------------------------------------------------
    chroma = np.asarray(
        chroma,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Validate extracted matrix
    # ---------------------------------------------------------
    if chroma.ndim != 2:
        raise ChromaExtractionError(
            "Expected a two-dimensional chroma feature matrix, "
            f"got shape {chroma.shape}."
        )

    if chroma.shape[0] != settings.n_chroma:
        raise ChromaExtractionError(
            "Unexpected number of chroma bins: "
            f"expected {settings.n_chroma}, "
            f"got {chroma.shape[0]}."
        )

    if chroma.shape[1] == 0:
        raise ChromaExtractionError(
            "Chroma extraction produced zero frames."
        )

    if not np.all(np.isfinite(chroma)):
        raise ChromaExtractionError(
            "Chroma features contain NaN or infinite values."
        )

    # ---------------------------------------------------------
    # Optional normalization
    # ---------------------------------------------------------
    if settings.normalize:
        mean = np.mean(chroma)
        std = np.std(chroma)

        chroma = (
            chroma - mean
        ) / (
            std + 1e-8
        )

    # ---------------------------------------------------------
    # Final numerical validation
    # ---------------------------------------------------------
    if not np.all(np.isfinite(chroma)):
        raise ChromaExtractionError(
            "Normalized chroma features contain NaN or infinite values."
        )

    # ---------------------------------------------------------
    # Return contiguous float32 matrix
    # ---------------------------------------------------------
    return np.ascontiguousarray(
        chroma,
        dtype=np.float32,
    )


__all__ = [
    "ChromaConfig",
    "ChromaExtractionError",
    "extract_chroma",
]