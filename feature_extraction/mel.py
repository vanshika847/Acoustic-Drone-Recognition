"""Mel spectrogram feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray

from feature_extraction.mfcc import MFCCExtractionError, _validate_waveform


@dataclass(frozen=True, slots=True)
class MelSpectrogramConfig:
    """Configuration for mel spectrogram extraction.

    Attributes:
        n_mels: Number of mel bands.
        n_fft: FFT window size in samples.
        hop_length: Number of samples between successive frames.
        power: Exponent applied to the magnitude spectrogram.
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
        if self.n_mels <= 0:
            raise MFCCExtractionError("n_mels must be greater than zero.")
        if self.n_fft <= 0:
            raise MFCCExtractionError("n_fft must be greater than zero.")
        if self.hop_length <= 0:
            raise MFCCExtractionError("hop_length must be greater than zero.")
        if self.power <= 0.0:
            raise MFCCExtractionError("power must be greater than zero.")
        if self.win_length <= 0:
            raise MFCCExtractionError("win_length must be greater than zero.")
        if self.fmin < 0:
            raise MFCCExtractionError("fmin must be non-negative.")
        if self.fmax is not None and self.fmax <= self.fmin:
            raise MFCCExtractionError(
                "fmax must be greater than fmin."
            )
        if self.win_length > self.n_fft:
            raise MFCCExtractionError(
                "win_length cannot be greater than n_fft."
          )


def extract_mel_spectrogram(
    waveform: NDArray[np.floating],
    sample_rate: int,
    config: MelSpectrogramConfig | None = None,
) -> NDArray[np.float32]:
    """Extract a power mel spectrogram from a mono waveform.

    Args:
        waveform: One-dimensional mono audio samples.
        sample_rate: Waveform sample rate in Hertz.
        config: Mel spectrogram parameters. Defaults to :class:`MelSpectrogramConfig`.

    Returns:
        Mel spectrogram with shape ``(n_mels, n_frames)`` as float32.

    Raises:
        MFCCExtractionError: If the waveform or sample rate is invalid.
    """

    settings = config or MelSpectrogramConfig()
    audio = _validate_waveform(waveform)
    if sample_rate <= 0:
        raise MFCCExtractionError("sample_rate must be greater than zero.")

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
        fmax=settings.fmax,
        power=settings.power,
)
    if settings.log_scale:
        mel_spectrogram = librosa.power_to_db(
            mel_spectrogram,
            ref=1.0
        )
    if settings.normalize:
        mean = np.mean(mel_spectrogram)

        std = np.std(mel_spectrogram)

        mel_spectrogram = (
            mel_spectrogram - mean
        ) / (
            std + 1e-8
        )

    return np.asarray(mel_spectrogram, dtype=np.float32)