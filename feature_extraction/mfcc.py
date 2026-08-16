"""MFCC feature extraction for model-ready mono audio.

Purpose
-------
Extract Mel-Frequency Cepstral Coefficients (MFCCs) from a validated mono
waveform, optionally including first- and second-order temporal derivatives.

The extractor is designed for the project's standardized audio pipeline:

    sample rate : 16 kHz
    audio       : mono
    segment     : 4 seconds

Output
------
When ``include_delta=True`` and ``n_mfcc=40``, the output contains:

    40 static MFCC coefficients
    40 delta coefficients
    40 delta-delta coefficients

for a total of 120 feature channels.

Output shape
------------
    (feature_channels, time_frames)

For the default configuration:

    (120, n_frames)

The exact number of time frames is determined by librosa's STFT framing
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray


class MFCCExtractionError(ValueError):
    """Raised when MFCC extraction receives invalid input or configuration."""


@dataclass(frozen=True, slots=True)
class MFCCConfig:
    """Configuration for MFCC feature extraction.

    Attributes:
        n_mfcc:
            Number of static MFCC coefficients.

        n_mels:
            Number of Mel filter-bank channels used before MFCC calculation.

        n_fft:
            FFT size in samples.

        hop_length:
            Number of samples between consecutive analysis frames.

        win_length:
            Length of the analysis window in samples.

        window:
            Window function used by the STFT.

        center:
            Whether librosa centers STFT frames by padding the waveform.

        fmin:
            Minimum frequency considered by the Mel filter bank.

        fmax:
            Maximum frequency considered by the Mel filter bank.
            ``None`` uses the Nyquist frequency.

        include_delta:
            Whether to append first- and second-order MFCC derivatives.

        normalize:
            Whether to standardize every feature channel across time.

        expected_sample_rate:
            Expected project sample rate. Set to ``None`` to disable the
            explicit project-level sample-rate check.
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
    expected_sample_rate: int | None = 16_000

    def __post_init__(self) -> None:
        """Validate static MFCC configuration values."""

        if self.n_mfcc <= 0:
            raise MFCCExtractionError(
                "n_mfcc must be greater than zero."
            )

        if self.n_mels <= 0:
            raise MFCCExtractionError(
                "n_mels must be greater than zero."
            )

        if self.n_mfcc > self.n_mels:
            raise MFCCExtractionError(
                "n_mfcc cannot be greater than n_mels."
            )

        if self.n_fft <= 0:
            raise MFCCExtractionError(
                "n_fft must be greater than zero."
            )

        if self.hop_length <= 0:
            raise MFCCExtractionError(
                "hop_length must be greater than zero."
            )

        if self.win_length <= 0:
            raise MFCCExtractionError(
                "win_length must be greater than zero."
            )

        if self.win_length > self.n_fft:
            raise MFCCExtractionError(
                "win_length cannot be greater than n_fft."
            )

        if self.fmin < 0.0:
            raise MFCCExtractionError(
                "fmin cannot be negative."
            )

        if self.fmax is not None and self.fmax <= self.fmin:
            raise MFCCExtractionError(
                "fmax must be greater than fmin when specified."
            )

        if (
            self.expected_sample_rate is not None
            and self.expected_sample_rate <= 0
        ):
            raise MFCCExtractionError(
                "expected_sample_rate must be greater than zero."
            )


def extract_mfcc(
    waveform: NDArray[np.floating],
    sample_rate: int,
    config: MFCCConfig | None = None,
) -> NDArray[np.float32]:
    """Extract MFCC features from a mono waveform.

    Processing:

        waveform
            ↓
        validation
            ↓
        MFCC extraction
            ↓
        optional delta
            ↓
        optional delta-delta
            ↓
        optional normalization
            ↓
        float32 feature matrix

    Args:
        waveform:
            One-dimensional mono audio samples.

        sample_rate:
            Sampling rate of the waveform in Hertz.

        config:
            MFCC configuration. If omitted, :class:`MFCCConfig` is used.

    Returns:
        MFCC feature matrix with shape:

            (feature_channels, time_frames)

        With the default configuration:

            (120, time_frames)

    Raises:
        MFCCExtractionError:
            If the waveform, sample rate, or configuration is invalid.
    """

    settings = config or MFCCConfig()

    _validate_sample_rate(
        sample_rate=sample_rate,
        config=settings,
    )

    audio = _validate_waveform(waveform)

    _validate_frequency_limits(
        sample_rate=sample_rate,
        config=settings,
    )

    try:
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
    except Exception as error:
        raise MFCCExtractionError(
            f"MFCC extraction failed: {error}"
        ) from error

    if mfcc.ndim != 2 or mfcc.size == 0:
        raise MFCCExtractionError(
            f"MFCC extraction produced an invalid shape: {mfcc.shape}."
        )

    if not np.all(np.isfinite(mfcc)):
        raise MFCCExtractionError(
            "MFCC extraction produced non-finite values."
        )

    if settings.include_delta:
        mfcc = _append_temporal_derivatives(mfcc)

    if settings.normalize:
        mfcc = _normalize_features(mfcc)

    if not np.all(np.isfinite(mfcc)):
        raise MFCCExtractionError(
            "Final MFCC features contain non-finite values."
        )

    return np.asarray(mfcc, dtype=np.float32)


def _append_temporal_derivatives(
    mfcc: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Append delta and delta-delta features to static MFCCs.

    Args:
        mfcc:
            Static MFCC matrix with shape ``(n_mfcc, n_frames)``.

    Returns:
        Concatenated matrix:

            [MFCC]
            [Delta]
            [Delta-Delta]

        with shape ``(3 * n_mfcc, n_frames)``.

    Raises:
        MFCCExtractionError:
            If there are insufficient frames for librosa's delta operation.
    """

    if mfcc.ndim != 2:
        raise MFCCExtractionError(
            f"Expected a two-dimensional MFCC matrix, got {mfcc.shape}."
        )

    n_frames = mfcc.shape[1]

    # librosa's default delta width is 9 frames.
    # A smaller odd width is selected for very short segments so that
    # feature extraction remains valid rather than failing on short input.
    width = _select_delta_width(n_frames)

    try:
        delta = librosa.feature.delta(
            mfcc,
            order=1,
            width=width,
            mode="nearest",
        )

        delta2 = librosa.feature.delta(
            mfcc,
            order=2,
            width=width,
            mode="nearest",
        )
    except Exception as error:
        raise MFCCExtractionError(
            f"MFCC delta extraction failed for {n_frames} frames: {error}"
        ) from error

    combined = np.concatenate(
        (
            mfcc,
            delta,
            delta2,
        ),
        axis=0,
    )

    return np.asarray(combined, dtype=np.float32)


def _select_delta_width(n_frames: int) -> int:
    """Select a valid odd delta window for the available frame count.

    Args:
        n_frames:
            Number of time frames in the MFCC matrix.

    Returns:
        Odd delta width suitable for librosa.

    Raises:
        MFCCExtractionError:
            If fewer than three frames are available.
    """

    if n_frames < 3:
        raise MFCCExtractionError(
            "At least 3 time frames are required for delta features."
        )

    # Use librosa's usual width of 9 whenever possible.
    if n_frames >= 9:
        return 9

    # For short inputs, use the largest valid odd width.
    width = n_frames if n_frames % 2 == 1 else n_frames - 1

    if width < 3:
        raise MFCCExtractionError(
            "Insufficient time frames for delta feature extraction."
        )

    return width


def _normalize_features(
    features: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Standardize each feature channel independently across time.

    Args:
        features:
            Feature matrix with shape ``(channels, time_frames)``.

    Returns:
        Float32 matrix where each channel is approximately zero mean and
        unit variance.

    Notes:
        A small epsilon is added to the standard deviation so constant
        channels do not cause division-by-zero.
    """

    mean = np.mean(
        features,
        axis=1,
        keepdims=True,
        dtype=np.float64,
    )

    std = np.std(
        features,
        axis=1,
        keepdims=True,
        dtype=np.float64,
    )

    normalized = (
        np.asarray(features, dtype=np.float64) - mean
    ) / (std + 1e-8)

    return np.asarray(normalized, dtype=np.float32)


def _validate_sample_rate(
    sample_rate: int,
    config: MFCCConfig,
) -> None:
    """Validate the supplied sample rate."""

    if not isinstance(sample_rate, (int, np.integer)):
        raise MFCCExtractionError(
            f"sample_rate must be an integer, got {type(sample_rate).__name__}."
        )

    if sample_rate <= 0:
        raise MFCCExtractionError(
            "sample_rate must be greater than zero."
        )

    if (
        config.expected_sample_rate is not None
        and sample_rate != config.expected_sample_rate
    ):
        raise MFCCExtractionError(
            "Unexpected sample rate: "
            f"received {sample_rate} Hz, expected "
            f"{config.expected_sample_rate} Hz."
        )


def _validate_frequency_limits(
    sample_rate: int,
    config: MFCCConfig,
) -> None:
    """Validate MFCC frequency limits against the Nyquist frequency."""

    nyquist = sample_rate / 2.0

    if config.fmin >= nyquist:
        raise MFCCExtractionError(
            f"fmin ({config.fmin} Hz) must be below the Nyquist "
            f"frequency ({nyquist} Hz)."
        )

    if config.fmax is not None and config.fmax > nyquist:
        raise MFCCExtractionError(
            f"fmax ({config.fmax} Hz) cannot exceed the Nyquist "
            f"frequency ({nyquist} Hz)."
        )


def _validate_waveform(
    waveform: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Validate and coerce a waveform for librosa feature extraction.

    Args:
        waveform:
            Candidate one-dimensional mono waveform.

    Returns:
        Finite float32 one-dimensional waveform.

    Raises:
        MFCCExtractionError:
            If the waveform is empty, non-numeric, non-finite, or not mono.
    """

    try:
        audio = np.asarray(waveform, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise MFCCExtractionError(
            "Waveform could not be converted to float32."
        ) from error

    if audio.ndim != 1:
        raise MFCCExtractionError(
            "Expected a one-dimensional mono waveform, "
            f"received shape {audio.shape}."
        )

    if audio.size == 0:
        raise MFCCExtractionError(
            "Waveform must contain at least one sample."
        )

    if not np.all(np.isfinite(audio)):
        raise MFCCExtractionError(
            "Waveform contains NaN or infinite values."
        )

    return np.ascontiguousarray(audio, dtype=np.float32)


__all__ = [
    "MFCCConfig",
    "MFCCExtractionError",
    "extract_mfcc",
]