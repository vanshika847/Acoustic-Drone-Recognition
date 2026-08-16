"""Mel spectrogram feature extraction.

This module converts a mono audio waveform into a mel-scaled spectrogram
suitable for downstream machine-learning feature extraction.

The extraction pipeline:
    1. Validate the waveform and sample rate.
    2. Compute a power spectrogram.
    3. Project the spectrogram onto the mel scale.
    4. Optionally convert power values to decibels.
    5. Optionally standardize the resulting feature matrix.

Output shape:
    (n_mels, n_frames)
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


class MelSpectrogramExtractionError(MFCCExtractionError):
    """Raised when mel spectrogram extraction receives invalid input."""


@dataclass(frozen=True, slots=True)
class MelSpectrogramConfig:
    """Configuration for mel spectrogram extraction.

    Attributes:
        n_mels:
            Number of mel-frequency bands.

        n_fft:
            FFT window size in samples.

        hop_length:
            Number of samples between successive frames.

        win_length:
            Length of the analysis window in samples.

        window:
            Window function used for the STFT.

        center:
            Whether the input waveform is padded so that frames are centered
            around their corresponding samples.

        fmin:
            Minimum frequency included in the mel filter bank.

        fmax:
            Maximum frequency included in the mel filter bank. If None,
            the Nyquist frequency is used.

        power:
            Exponent applied to the magnitude spectrogram.
            ``2.0`` produces a power spectrogram.

        log_scale:
            Whether to convert the mel power spectrogram to decibels.

        normalize:
            Whether to standardize the complete mel feature matrix.
    """

    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 1024
    window: str = "hann"
    center: bool = True
    fmin: float = 20.0
    fmax: float | None = None
    power: float = 2.0
    log_scale: bool = True
    normalize: bool = True

    def __post_init__(self) -> None:
        """Validate mel spectrogram configuration."""

        if self.n_mels <= 0:
            raise MelSpectrogramExtractionError(
                "n_mels must be greater than zero."
            )

        if self.n_fft <= 0:
            raise MelSpectrogramExtractionError(
                "n_fft must be greater than zero."
            )

        if self.hop_length <= 0:
            raise MelSpectrogramExtractionError(
                "hop_length must be greater than zero."
            )

        if self.win_length <= 0:
            raise MelSpectrogramExtractionError(
                "win_length must be greater than zero."
            )

        if self.win_length > self.n_fft:
            raise MelSpectrogramExtractionError(
                "win_length cannot be greater than n_fft."
            )

        if self.power <= 0.0:
            raise MelSpectrogramExtractionError(
                "power must be greater than zero."
            )

        if self.fmin < 0.0:
            raise MelSpectrogramExtractionError(
                "fmin must be non-negative."
            )

        if self.fmax is not None:
            if self.fmax <= self.fmin:
                raise MelSpectrogramExtractionError(
                    "fmax must be greater than fmin."
                )


def extract_mel_spectrogram(
    waveform: NDArray[np.floating],
    sample_rate: int,
    config: MelSpectrogramConfig | None = None,
) -> NDArray[np.float32]:
    """Extract a mel spectrogram from a mono waveform.

    Args:
        waveform:
            One-dimensional mono audio samples.

        sample_rate:
            Waveform sample rate in Hertz.

        config:
            Mel spectrogram parameters. If omitted, the default
            :class:`MelSpectrogramConfig` is used.

    Returns:
        Mel spectrogram with shape ``(n_mels, n_frames)`` as float32.

    Raises:
        MelSpectrogramExtractionError:
            If the waveform, sample rate, or configuration is invalid.
    """

    settings = config or MelSpectrogramConfig()

    # Validate and convert waveform.
    try:
        audio = _validate_waveform(waveform)
    except MFCCExtractionError as exc:
        raise MelSpectrogramExtractionError(str(exc)) from exc

    # Validate sample rate.
    if sample_rate <= 0:
        raise MelSpectrogramExtractionError(
            "sample_rate must be greater than zero."
        )

    # Nyquist frequency is the highest valid frequency.
    nyquist_frequency = sample_rate / 2.0

    # Validate frequency limits against the sample rate.
    if settings.fmin >= nyquist_frequency:
        raise MelSpectrogramExtractionError(
            f"fmin ({settings.fmin}) must be below the Nyquist "
            f"frequency ({nyquist_frequency})."
        )

    effective_fmax = settings.fmax

    if effective_fmax is not None:
        if effective_fmax > nyquist_frequency:
            raise MelSpectrogramExtractionError(
                f"fmax ({effective_fmax}) cannot exceed the Nyquist "
                f"frequency ({nyquist_frequency})."
            )

    # Compute mel spectrogram.
    try:
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=settings.n_mels,
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
            win_length=settings.win_length,
            window=settings.window,
            center=settings.center,
            fmin=settings.fmin,
            fmax=effective_fmax,
            power=settings.power,
        )
    except Exception as exc:
        raise MelSpectrogramExtractionError(
            f"Failed to compute mel spectrogram: {exc}"
        ) from exc

    # Convert power values to decibels.
    if settings.log_scale:
        mel_spectrogram = librosa.power_to_db(
            mel_spectrogram,
            ref=1.0,
        )

    # Standardize feature values.
    if settings.normalize:
        mean = np.mean(mel_spectrogram)
        std = np.std(mel_spectrogram)

        mel_spectrogram = (
            mel_spectrogram - mean
        ) / (
            std + 1e-8
        )

    # Final safety check.
    if not np.all(np.isfinite(mel_spectrogram)):
        raise MelSpectrogramExtractionError(
            "Mel spectrogram contains NaN or infinite values."
        )

    return np.asarray(
        mel_spectrogram,
        dtype=np.float32,
    )


__all__ = [
    "MelSpectrogramConfig",
    "MelSpectrogramExtractionError",
    "extract_mel_spectrogram",
]