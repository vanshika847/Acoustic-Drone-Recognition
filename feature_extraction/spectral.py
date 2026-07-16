"""Spectral feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray

from feature_extraction.mfcc import MFCCExtractionError, _validate_waveform

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
        n_fft: FFT window size in samples.
        hop_length: Number of samples between successive frames.
        roll_percent: Roll-off percentage for spectral roll-off.
        n_bands: Number of frequency bands for spectral contrast.
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
        if self.n_fft <= 0:
            raise MFCCExtractionError("n_fft must be greater than zero.")
        if self.hop_length <= 0:
            raise MFCCExtractionError("hop_length must be greater than zero.")
        
        if self.n_bands <= 0:
            raise MFCCExtractionError("n_bands must be greater than zero.")
        if self.win_length <= 0:
            raise MFCCExtractionError(
                "win_length must be greater than zero."
            )

        if self.win_length > self.n_fft:
            raise MFCCExtractionError(
                "win_length cannot exceed n_fft."
            )

        if self.fmin < 0:
            raise MFCCExtractionError(
                "fmin must be non-negative."
            )

        if self.fmax is not None and self.fmax <= self.fmin:
            raise MFCCExtractionError(
                "fmax must be greater than fmin."
            )


def extract_spectral_features(
    waveform: NDArray[np.floating],
    sample_rate: int,
    config: SpectralConfig | None = None,
) -> NDArray[np.float32]:
    """Extract frame-wise spectral descriptors from a mono waveform.

    The returned matrix stacks centroid, bandwidth, roll-off, and contrast rows
    in the order defined by :data:`SPECTRAL_FEATURE_NAMES`.

    Args:
        waveform: One-dimensional mono audio samples.
        sample_rate: Waveform sample rate in Hertz.
        config: Spectral parameters. Defaults to :class:`SpectralConfig`.

    Returns:
        Spectral feature matrix with shape ``(n_features, n_frames)`` as float32.

    Raises:
        MFCCExtractionError: If the waveform or sample rate is invalid.
    """

    settings = config or SpectralConfig()
    audio = _validate_waveform(waveform)
    if sample_rate <= 0:
        raise MFCCExtractionError("sample_rate must be greater than zero.")
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        win_length=settings.win_length,
        window=settings.window,
        center=settings.center,
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        win_length=settings.win_length,
        window=settings.window,
        center=settings.center,
    )
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
    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        win_length=settings.win_length,
    )
    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sample_rate,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        n_bands=settings.n_bands,
        fmin=settings.fmin,
        win_length=settings.win_length,
        window=settings.window,
        center=settings.center,
    )


    stacked = np.vstack(
        (
        np.asarray(centroid, dtype=np.float32),
        np.asarray(bandwidth, dtype=np.float32),
        np.asarray(rolloff85, dtype=np.float32),
        np.asarray(rolloff95, dtype=np.float32),
        np.asarray(flatness, dtype=np.float32),
        np.asarray(contrast, dtype=np.float32),
    )
)
    if settings.normalize:
        mean = np.mean(stacked)

        std = np.std(stacked)

        stacked = (
            stacked - mean
        ) / (
            std + 1e-8
                )
        
        
    return np.ascontiguousarray(stacked, dtype=np.float32)