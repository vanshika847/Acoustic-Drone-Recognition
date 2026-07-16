"""Zero crossing rate feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray

from feature_extraction.mfcc import MFCCExtractionError, _validate_waveform


@dataclass(frozen=True, slots=True)
class ZCRConfig:
    """Configuration for zero crossing rate extraction.

    Attributes:
        frame_length: Analysis frame length in samples.
        hop_length: Number of samples between successive frames.
    """

    frame_length: int = 2048

    hop_length: int = 512

    center: bool = True

    threshold: float = 0.0

    zero_pos: bool = True

    normalize: bool = True

    def __post_init__(self) -> None:
        if self.frame_length <= 0:
            raise MFCCExtractionError("frame_length must be greater than zero.")
        if self.hop_length <= 0:
            raise MFCCExtractionError("hop_length must be greater than zero.")
        if self.threshold < 0:
            raise MFCCExtractionError("threshold must be non-negative.")
    


def extract_zcr(
    waveform: NDArray[np.floating],
    sample_rate: int,
    config: ZCRConfig | None = None,
) -> NDArray[np.float32]:
    """Extract frame-wise zero crossing rate from a mono waveform.

    Args:
        waveform: One-dimensional mono audio samples.
        sample_rate: Waveform sample rate in Hertz. Retained for API consistency.
        config: ZCR parameters. Defaults to :class:`ZCRConfig`.

    Returns:
        Zero crossing rate matrix with shape ``(1, n_frames)`` as float32.

    Raises:
        MFCCExtractionError: If the waveform is invalid.
    """

    if sample_rate <= 0:
        raise MFCCExtractionError(
            "sample_rate must be greater than zero."
        )
    settings = config or ZCRConfig()
    audio = _validate_waveform(waveform)

    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=settings.frame_length,
        hop_length=settings.hop_length,
        center=settings.center,
        threshold=settings.threshold,
        zero_pos=settings.zero_pos,
        
    )

    if settings.normalize:
        mean = np.mean(zcr)

        std = np.std(zcr)

        zcr = (
            zcr - mean
        ) / (
            std + 1e-8
        )

    return np.asarray(zcr, dtype=np.float32)