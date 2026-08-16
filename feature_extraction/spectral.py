"""Spectral feature extraction.

This module extracts frame-wise spectral descriptors from a mono waveform.

Features:
    - Spectral centroid
    - Spectral bandwidth
    - Spectral rolloff at 85%
    - Spectral rolloff at 95%
    - Spectral flatness
    - Spectral contrast bands 0 through 6

The returned feature matrix has shape:

    (n_features, n_frames)

with features ordered according to ``SPECTRAL_FEATURE_NAMES``.
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


class SpectralExtractionError(MFCCExtractionError):
    """Raised when spectral feature extraction fails."""


SPECTRAL_FEATURE_NAMES = (
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff_85",
    "spectral_rolloff_95",
    "spectral_flatness",
    "spectral_contrast_0",
    "spectral_contrast_1",
    "spectral_contrast_2",
    "spectral_contrast_3",
    "spectral_contrast_4",
    "spectral_contrast_5",
    "spectral_contrast_6",
)


@dataclass(frozen=True, slots=True)
class SpectralConfig:
    """Configuration for spectral feature extraction.

    Attributes:
        n_fft:
            FFT window size in samples.

        win_length:
            Length of the analysis window in samples.

        window:
            Window function used for STFT-based features.

        center:
            Whether frames are centered by padding the waveform.

        hop_length:
            Number of samples between successive frames.

        n_bands:
            Number of spectral-contrast frequency bands.

        fmin:
            Minimum frequency used for spectral contrast.

        fmax:
            Maximum frequency used by the feature configuration.
            Currently used for validation and future compatibility.

        normalize:
            Whether to standardize the complete spectral feature matrix.
    """

    n_fft: int = 2048
    win_length: int = 2048
    window: str = "hann"
    center: bool = True
    hop_length: int = 512

    n_bands: int = 6
    fmin: float = 20.0
    fmax: float | None = None

    normalize: bool = True

    def __post_init__(self) -> None:
        """Validate spectral extraction settings."""

        if self.n_fft <= 0:
            raise SpectralExtractionError(
                "n_fft must be greater than zero."
            )

        if self.hop_length <= 0:
            raise SpectralExtractionError(
                "hop_length must be greater than zero."
            )

        if self.win_length <= 0:
            raise SpectralExtractionError(
                "win_length must be greater than zero."
            )

        if self.win_length > self.n_fft:
            raise SpectralExtractionError(
                "win_length cannot exceed n_fft."
            )

        if self.n_bands <= 0:
            raise SpectralExtractionError(
                "n_bands must be greater than zero."
            )

        if self.fmin < 0.0:
            raise SpectralExtractionError(
                "fmin must be non-negative."
            )

        if self.fmax is not None and self.fmax <= self.fmin:
            raise SpectralExtractionError(
                "fmax must be greater than fmin."
            )


def extract_spectral_features(
    waveform: NDArray[np.floating],
    sample_rate: int,
    config: SpectralConfig | None = None,
) -> NDArray[np.float32]:
    """Extract frame-wise spectral descriptors.

    The returned matrix contains features in the following order:

        1. Spectral centroid
        2. Spectral bandwidth
        3. Spectral rolloff at 85%
        4. Spectral rolloff at 95%
        5. Spectral flatness
        6. Spectral contrast band 0
        7. Spectral contrast band 1
        8. Spectral contrast band 2
        9. Spectral contrast band 3
        10. Spectral contrast band 4
        11. Spectral contrast band 5
        12. Spectral contrast band 6

    Args:
        waveform:
            One-dimensional mono audio samples.

        sample_rate:
            Waveform sample rate in Hertz.

        config:
            Spectral extraction parameters. If omitted, the default
            :class:`SpectralConfig` is used.

    Returns:
        Spectral feature matrix with shape
        ``(12, n_frames)`` as float32.

    Raises:
        SpectralExtractionError:
            If the waveform, sample rate, or spectral configuration
            is invalid.
    """

    settings = config or SpectralConfig()

    # Validate waveform.
    try:
        audio = _validate_waveform(waveform)
    except MFCCExtractionError as exc:
        raise SpectralExtractionError(str(exc)) from exc

    # Validate sample rate.
    if sample_rate <= 0:
        raise SpectralExtractionError(
            "sample_rate must be greater than zero."
        )

    nyquist_frequency = sample_rate / 2.0

    # Validate fmin against Nyquist.
    if settings.fmin >= nyquist_frequency:
        raise SpectralExtractionError(
            f"fmin ({settings.fmin}) must be below the Nyquist "
            f"frequency ({nyquist_frequency})."
        )

    # Validate optional fmax.
    if settings.fmax is not None:
        if settings.fmax > nyquist_frequency:
            raise SpectralExtractionError(
                f"fmax ({settings.fmax}) cannot exceed the Nyquist "
                f"frequency ({nyquist_frequency})."
            )

    try:
        # ---------------------------------------------------------
        # 1. Spectral Centroid
        # ---------------------------------------------------------
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sample_rate,
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
            win_length=settings.win_length,
            window=settings.window,
            center=settings.center,
        )

        # ---------------------------------------------------------
        # 2. Spectral Bandwidth
        # ---------------------------------------------------------
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sample_rate,
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
            win_length=settings.win_length,
            window=settings.window,
            center=settings.center,
        )

        # ---------------------------------------------------------
        # 3. Spectral Rolloff - 85%
        # ---------------------------------------------------------
        rolloff85 = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sample_rate,
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
            win_length=settings.win_length,
            window=settings.window,
            center=settings.center,
            roll_percent=0.85,
        )

        # ---------------------------------------------------------
        # 4. Spectral Rolloff - 95%
        # ---------------------------------------------------------
        rolloff95 = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sample_rate,
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
            win_length=settings.win_length,
            window=settings.window,
            center=settings.center,
            roll_percent=0.95,
        )

        # ---------------------------------------------------------
        # 5. Spectral Flatness
        # ---------------------------------------------------------
        flatness = librosa.feature.spectral_flatness(
            y=audio,
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
            win_length=settings.win_length,
            center=settings.center,
        )

        # ---------------------------------------------------------
        # 6-12. Spectral Contrast
        # ---------------------------------------------------------
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=sample_rate,
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
            win_length=settings.win_length,
            window=settings.window,
            center=settings.center,
            n_bands=settings.n_bands,
            fmin=settings.fmin,
        )

    except Exception as exc:
        raise SpectralExtractionError(
            f"Failed to compute spectral features: {exc}"
        ) from exc

    # Convert every feature to float32 before stacking.
    centroid = np.asarray(centroid, dtype=np.float32)
    bandwidth = np.asarray(bandwidth, dtype=np.float32)
    rolloff85 = np.asarray(rolloff85, dtype=np.float32)
    rolloff95 = np.asarray(rolloff95, dtype=np.float32)
    flatness = np.asarray(flatness, dtype=np.float32)
    contrast = np.asarray(contrast, dtype=np.float32)

    # Stack into one feature matrix.
    stacked = np.vstack(
        (
            centroid,
            bandwidth,
            rolloff85,
            rolloff95,
            flatness,
            contrast,
        )
    )

    # Verify expected feature count.
    if stacked.shape[0] != len(SPECTRAL_FEATURE_NAMES):
        raise SpectralExtractionError(
            "Unexpected number of spectral features: "
            f"expected {len(SPECTRAL_FEATURE_NAMES)}, "
            f"got {stacked.shape[0]}."
        )

    # Verify numerical validity before normalization.
    if not np.all(np.isfinite(stacked)):
        raise SpectralExtractionError(
            "Spectral features contain NaN or infinite values."
        )

    # -------------------------------------------------------------
    # Optional normalization
    # -------------------------------------------------------------
    if settings.normalize:
        mean = np.mean(stacked)
        std = np.std(stacked)

        stacked = (
            stacked - mean
        ) / (
            std + 1e-8
        )

    # Final numerical safety check.
    if not np.all(np.isfinite(stacked)):
        raise SpectralExtractionError(
            "Normalized spectral features contain NaN or infinite values."
        )

    return np.ascontiguousarray(
        stacked,
        dtype=np.float32,
    )


__all__ = [
    "SPECTRAL_FEATURE_NAMES",
    "SpectralConfig",
    "SpectralExtractionError",
    "extract_spectral_features",
]