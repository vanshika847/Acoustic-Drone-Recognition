"""Frame-wise RMS energy feature extraction.

This module extracts frame-wise Root Mean Square (RMS) energy
from a mono audio waveform.

RMS energy represents the average signal magnitude within each
analysis frame and provides information about the energy/intensity
of the audio signal.

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


class EnergyExtractionError(MFCCExtractionError):
    """Raised when RMS energy extraction fails."""


@dataclass(frozen=True, slots=True)
class EnergyConfig:
    """Configuration for frame-wise RMS energy extraction.

    Attributes:
        frame_length:
            Length of each analysis frame in samples.

        hop_length:
            Number of samples between successive frames.

        center:
            Whether frames are centered by padding the waveform.

        pad_mode:
            Padding mode used when ``center=True``.

        normalize:
            Whether to standardize the resulting RMS energy matrix.
    """

    frame_length: int = 2048
    hop_length: int = 512
    center: bool = True
    pad_mode: str = "constant"
    normalize: bool = True

    def __post_init__(self) -> None:
        """Validate RMS energy extraction settings."""

        if self.frame_length <= 0:
            raise EnergyExtractionError(
                "frame_length must be greater than zero."
            )

        if self.hop_length <= 0:
            raise EnergyExtractionError(
                "hop_length must be greater than zero."
            )

        if self.pad_mode not in (
            "constant",
            "reflect",
            "edge",
        ):
            raise EnergyExtractionError(
                "Unsupported pad_mode. "
                "Use 'constant', 'reflect', or 'edge'."
            )


def extract_energy(
    waveform: NDArray[np.floating],
    sample_rate: int,
    config: EnergyConfig | None = None,
) -> NDArray[np.float32]:
    """Extract frame-wise RMS energy from a mono waveform.

    Args:
        waveform:
            One-dimensional mono audio samples.

        sample_rate:
            Waveform sample rate in Hertz.

            The value is retained for API consistency with the
            other feature extraction functions.

        config:
            Energy extraction parameters. If omitted, the default
            :class:`EnergyConfig` is used.

    Returns:
        RMS energy matrix with shape
        ``(1, n_frames)`` as float32.

    Raises:
        EnergyExtractionError:
            If the waveform, sample rate, configuration, or
            extracted feature matrix is invalid.
    """

    settings = config or EnergyConfig()

    # ---------------------------------------------------------
    # Validate sample rate
    # ---------------------------------------------------------
    if sample_rate <= 0:
        raise EnergyExtractionError(
            "sample_rate must be greater than zero."
        )

    # ---------------------------------------------------------
    # Validate waveform
    # ---------------------------------------------------------
    try:
        audio = _validate_waveform(waveform)
    except MFCCExtractionError as exc:
        raise EnergyExtractionError(str(exc)) from exc

    # ---------------------------------------------------------
    # Validate waveform length
    # ---------------------------------------------------------
    if (
        audio.size < settings.frame_length
        and not settings.center
    ):
        raise EnergyExtractionError(
            "Waveform is shorter than frame_length "
            "while center=False."
        )

    # ---------------------------------------------------------
    # Extract RMS energy
    # ---------------------------------------------------------
    try:
        rms = librosa.feature.rms(
            y=audio,
            frame_length=settings.frame_length,
            hop_length=settings.hop_length,
            center=settings.center,
            pad_mode=settings.pad_mode,
        )

    except Exception as exc:
        raise EnergyExtractionError(
            f"Failed to compute RMS energy: {exc}"
        ) from exc

    # ---------------------------------------------------------
    # Convert to float32
    # ---------------------------------------------------------
    rms = np.asarray(
        rms,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Validate extracted features
    # ---------------------------------------------------------
    if rms.ndim != 2:
        raise EnergyExtractionError(
            "Expected a two-dimensional RMS energy matrix, "
            f"got shape {rms.shape}."
        )

    if rms.shape[0] != 1:
        raise EnergyExtractionError(
            "Unexpected number of RMS energy rows: "
            f"expected 1, got {rms.shape[0]}."
        )

    if rms.shape[1] == 0:
        raise EnergyExtractionError(
            "RMS energy extraction produced zero frames."
        )

    if not np.all(np.isfinite(rms)):
        raise EnergyExtractionError(
            "RMS energy contains NaN or infinite values."
        )

    # ---------------------------------------------------------
    # Optional normalization
    # ---------------------------------------------------------
    if settings.normalize:
        mean = np.mean(rms)
        std = np.std(rms)

        rms = (
            rms - mean
        ) / (
            std + 1e-8
        )

    # ---------------------------------------------------------
    # Final numerical validation
    # ---------------------------------------------------------
    if not np.all(np.isfinite(rms)):
        raise EnergyExtractionError(
            "Normalized RMS energy contains NaN or infinite values."
        )

    # ---------------------------------------------------------
    # Return contiguous float32 matrix
    # ---------------------------------------------------------
    return np.ascontiguousarray(
        rms,
        dtype=np.float32,
    )


__all__ = [
    "EnergyConfig",
    "EnergyExtractionError",
    "extract_energy",
]