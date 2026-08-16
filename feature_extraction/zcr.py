"""Zero crossing rate feature extraction.

This module extracts frame-wise zero crossing rate (ZCR) features
from a mono audio waveform.

ZCR measures how frequently the audio waveform changes sign.
It can help distinguish characteristics such as noisy, percussive,
or tonal signals.

The returned feature matrix has shape:

    (1, n_frames)
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


class ZCRExtractionError(MFCCExtractionError):
    """Raised when zero crossing rate extraction fails."""


@dataclass(frozen=True, slots=True)
class ZCRConfig:
    """Configuration for zero crossing rate extraction.

    Attributes:
        frame_length:
            Length of each analysis frame in samples.

        hop_length:
            Number of samples between successive frames.

        center:
            Whether frames are centered by padding the waveform.

        threshold:
            Magnitude threshold below which samples are treated as
            zero when determining sign changes.

        zero_pos:
            Whether zero-valued samples are treated as positive.

        normalize:
            Whether to standardize the resulting ZCR matrix.
    """

    frame_length: int = 2048
    hop_length: int = 512
    center: bool = True
    threshold: float = 0.0
    zero_pos: bool = True
    normalize: bool = True

    def __post_init__(self) -> None:
        """Validate ZCR extraction settings."""

        if self.frame_length <= 0:
            raise ZCRExtractionError(
                "frame_length must be greater than zero."
            )

        if self.hop_length <= 0:
            raise ZCRExtractionError(
                "hop_length must be greater than zero."
            )

        if self.threshold < 0.0:
            raise ZCRExtractionError(
                "threshold must be non-negative."
            )

        if not np.isfinite(self.threshold):
            raise ZCRExtractionError(
                "threshold must be a finite value."
            )


def extract_zcr(
    waveform: NDArray[np.floating],
    sample_rate: int,
    config: ZCRConfig | None = None,
) -> NDArray[np.float32]:
    """Extract frame-wise zero crossing rate from a mono waveform.

    Args:
        waveform:
            One-dimensional mono audio samples.

        sample_rate:
            Waveform sample rate in Hertz.

            The value is retained for API consistency with the other
            feature extraction functions.

        config:
            ZCR extraction parameters. If omitted, the default
            :class:`ZCRConfig` is used.

    Returns:
        Zero crossing rate matrix with shape
        ``(1, n_frames)`` as float32.

    Raises:
        ZCRExtractionError:
            If the waveform, sample rate, configuration, or extracted
            feature matrix is invalid.
    """

    settings = config or ZCRConfig()

    # ---------------------------------------------------------
    # Validate sample rate
    # ---------------------------------------------------------
    if sample_rate <= 0:
        raise ZCRExtractionError(
            "sample_rate must be greater than zero."
        )

    # ---------------------------------------------------------
    # Validate waveform
    # ---------------------------------------------------------
    try:
        audio = _validate_waveform(waveform)
    except MFCCExtractionError as exc:
        raise ZCRExtractionError(str(exc)) from exc

    # ---------------------------------------------------------
    # Validate waveform length
    # ---------------------------------------------------------
    if (
        audio.size < settings.frame_length
        and not settings.center
    ):
        raise ZCRExtractionError(
            "Waveform is shorter than frame_length "
            "while center=False."
        )

    # ---------------------------------------------------------
    # Extract ZCR
    # ---------------------------------------------------------
    try:
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=settings.frame_length,
            hop_length=settings.hop_length,
            center=settings.center,
            threshold=settings.threshold,
            zero_pos=settings.zero_pos,
        )

    except Exception as exc:
        raise ZCRExtractionError(
            f"Failed to compute zero crossing rate: {exc}"
        ) from exc

    # ---------------------------------------------------------
    # Convert to float32
    # ---------------------------------------------------------
    zcr = np.asarray(
        zcr,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Validate extracted features
    # ---------------------------------------------------------
    if zcr.ndim != 2:
        raise ZCRExtractionError(
            "Expected a two-dimensional ZCR feature matrix, "
            f"got shape {zcr.shape}."
        )

    if zcr.shape[0] != 1:
        raise ZCRExtractionError(
            "Unexpected number of ZCR feature rows: "
            f"expected 1, got {zcr.shape[0]}."
        )

    if zcr.shape[1] == 0:
        raise ZCRExtractionError(
            "ZCR extraction produced zero frames."
        )

    if not np.all(np.isfinite(zcr)):
        raise ZCRExtractionError(
            "ZCR features contain NaN or infinite values."
        )

    # ---------------------------------------------------------
    # Optional normalization
    # ---------------------------------------------------------
    if settings.normalize:
        mean = np.mean(zcr)
        std = np.std(zcr)

        zcr = (
            zcr - mean
        ) / (
            std + 1e-8
        )

    # ---------------------------------------------------------
    # Final numerical validation
    # ---------------------------------------------------------
    if not np.all(np.isfinite(zcr)):
        raise ZCRExtractionError(
            "Normalized ZCR features contain NaN or infinite values."
        )

    # ---------------------------------------------------------
    # Return contiguous float32 matrix
    # ---------------------------------------------------------
    return np.ascontiguousarray(
        zcr,
        dtype=np.float32,
    )


__all__ = [
    "ZCRConfig",
    "ZCRExtractionError",
    "extract_zcr",
]