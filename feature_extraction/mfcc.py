"""MFCC feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray


class MFCCExtractionError(ValueError):
    """Raised when MFCC extraction receives invalid input."""


@dataclass(frozen=True, slots=True)
class MFCCConfig:
    """
    Configuration for MFCC extraction.
    """

    n_mfcc: int = 40

    n_mels: int = 128

    n_fft: int = 1024

    hop_length: int = 512

    win_length: int = 1024

    window: str = "hann"

    center: bool = True

    fmin: float = 20.0

    fmax: float | None = None

    include_delta: bool = True

    normalize: bool = True

    def __post_init__(self):

        if self.n_mfcc <= 0:
            raise MFCCExtractionError("n_mfcc must be greater than zero.")

        if self.n_mels <= 0:
            raise MFCCExtractionError("n_mels must be greater than zero.")

        if self.n_fft <= 0:
            raise MFCCExtractionError("n_fft must be greater than zero.")

        if self.hop_length <= 0:
            raise MFCCExtractionError("hop_length must be greater than zero.")

        if self.win_length <= 0:
            raise MFCCExtractionError("win_length must be greater than zero.")


def extract_mfcc(
    waveform: NDArray[np.floating],
    sample_rate: int,
    config: MFCCConfig | None = None,
) -> NDArray[np.float32]:
    """Extract MFCC features from a mono waveform.

    Args:
        waveform: One-dimensional mono audio samples.
        sample_rate: Waveform sample rate in Hertz.
        config: MFCC parameters. Defaults to :class:`MFCCConfig`.

    Returns:
        MFCC matrix with shape ``(n_mfcc, n_frames)`` as float32.

    Raises:
        MFCCExtractionError: If the waveform or sample rate is invalid.
    """

    settings = config or MFCCConfig()
    audio = _validate_waveform(waveform)
    if sample_rate <= 0:
        raise MFCCExtractionError("sample_rate must be greater than zero.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=settings.n_mfcc,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        win_length=settings.win_length,
        window=settings.window,
        center=settings.center,
        n_mels=settings.n_mels,
        fmin=settings.fmin,
        fmax=settings.fmax,
    )
    if settings.include_delta:
        delta = librosa.feature.delta(mfcc)

        delta2 = librosa.feature.delta(
            mfcc,
            order=2,
        )

        mfcc = np.concatenate(
            [
                mfcc,
                delta,
                delta2,
            ],
            axis=0,
        )
    if settings.normalize:
        mean = np.mean(
            mfcc,
            axis=1,
            keepdims=True
        )

        std = np.std(
            mfcc,
            axis=1,
            keepdims=True
        )

        mfcc = (mfcc - mean) / (std + 1e-8)

    return np.asarray(mfcc, dtype=np.float32)


def _validate_waveform(waveform: NDArray[np.floating]) -> NDArray[np.float32]:
    """Validate and coerce a waveform for librosa feature extraction.

    Args:
        waveform: Candidate mono audio samples.

    Returns:
        Finite float32 one-dimensional waveform.

    Raises:
        MFCCExtractionError: If the waveform is empty or non-finite.
    """

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise MFCCExtractionError(
            f"Expected a one-dimensional waveform, received shape {audio.shape}."
        )
    if audio.size == 0:
        raise MFCCExtractionError("Waveform must contain at least one sample.")
    if not np.all(np.isfinite(audio)):
        raise MFCCExtractionError("Waveform contains non-finite values.")
    return audio
    
    