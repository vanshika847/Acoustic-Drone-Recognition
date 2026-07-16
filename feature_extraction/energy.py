"""Frame-wise RMS energy feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray

from feature_extraction.mfcc import MFCCExtractionError, _validate_waveform


@dataclass(frozen=True, slots=True)
class EnergyConfig:
    """Configuration for frame-wise RMS energy extraction.

    Attributes:
        frame_length: Analysis frame length in samples.
        hop_length: Number of samples between successive frames.
    """

    frame_length: int = 2048

    hop_length: int = 512

    center: bool = True

    pad_mode: str = "constant"

    normalize: bool = True

    def __post_init__(self) -> None:
        if self.frame_length <= 0:
            raise MFCCExtractionError("frame_length must be greater than zero.")
        if self.hop_length <= 0:
            raise MFCCExtractionError("hop_length must be greater than zero.")
        if self.pad_mode not in (
            "constant",
            "reflect",
            "edge",
        ):
            raise MFCCExtractionError(
                "Unsupported pad_mode."
            )


def extract_energy(
    waveform: NDArray[np.floating],
    sample_rate: int,
    config: EnergyConfig | None = None,
) -> NDArray[np.float32]:
    """Extract frame-wise RMS energy from a mono waveform.

    Args:
        waveform: One-dimensional mono audio samples.
        sample_rate: Waveform sample rate in Hertz. Retained for API consistency.
        config: Energy parameters. Defaults to :class:`EnergyConfig`.

    Returns:
        RMS energy matrix with shape ``(1, n_frames)`` as float32.

    Raises:
        MFCCExtractionError: If the waveform is invalid.
    """

    if sample_rate <= 0:
        raise MFCCExtractionError(
            "sample_rate must be greater than zero."
        )
    settings = config or EnergyConfig()
    audio = _validate_waveform(waveform)

    rms = librosa.feature.rms(
        y=audio,
        frame_length=settings.frame_length,
        hop_length=settings.hop_length,
        center=settings.center,
        pad_mode=settings.pad_mode,
    )

    if settings.normalize:
        std = np.std(rms)

        if std > 1e-8:
            rms = (
                rms - np.mean(rms)
            ) / (
                std + 1e-8
            )

    return np.asarray(rms, dtype=np.float32)