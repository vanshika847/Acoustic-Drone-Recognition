"""
Reusable fixed-window segmentation for mono audio waveforms.

Purpose
-------
Convert a variable-length mono waveform into fixed-length clips suitable for
machine-learning feature extraction and batch training.

This module is pure:
- It does not read files.
- It does not write files.
- It does not modify labels.

Inputs
------
A one-dimensional finite NumPy waveform and a SegmentationConfig.

Outputs
-------
An ordered sequence of AudioSegment objects, each containing:
- segment index
- source start/end sample positions
- fixed-length waveform
- padding information

Algorithm
---------
Starting at sample zero, a fixed-size window advances by hop_samples.

Full windows are emitted directly.

A final partial window is emitted only when it contains at least
minimum_final_seconds of original audio. If padding is enabled, the remaining
samples are filled with zeros so that the output has the same fixed length
as every other segment.

The segmentation is deterministic and does not depend on filesystem state.

Example
-------
    import numpy as np

    config = SegmentationConfig(
        sample_rate=16_000,
        window_seconds=4.0,
        hop_seconds=2.0,
    )

    segmenter = AudioSegmenter(config)

    waveform = np.zeros(64_000, dtype=np.float32)

    segments = segmenter.segment(waveform)

    len(segments)
    8
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class AudioSegmentationError(ValueError):
    """Raised when audio or segmentation settings are invalid."""


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    """
    Parameters governing deterministic waveform segmentation.

    Attributes
    ----------
    sample_rate:
        Waveform sample rate in Hertz.

    window_seconds:
        Target duration of each emitted segment.

    hop_seconds:
        Time between adjacent segment starts.

        If None, the hop is equal to window_seconds, producing
        non-overlapping segments.

    minimum_final_seconds:
        Minimum amount of real source audio required for retaining
        a trailing partial segment.

    pad_final_segment:
        Whether a retained partial segment should be zero-padded
        to the full window length.
    """

    sample_rate: int
    window_seconds: float
    hop_seconds: float | None = None
    minimum_final_seconds: float = 1.0
    pad_final_segment: bool = True

    def __post_init__(self) -> None:
        """Validate segmentation settings."""

        effective_hop_seconds = (
            self.hop_seconds
            if self.hop_seconds is not None
            else self.window_seconds
        )

        if self.sample_rate <= 0:
            raise AudioSegmentationError(
                "sample_rate must be greater than zero."
            )

        if self.window_seconds <= 0.0:
            raise AudioSegmentationError(
                "window_seconds must be greater than zero."
            )

        if effective_hop_seconds <= 0.0:
            raise AudioSegmentationError(
                "hop_seconds must be greater than zero."
            )

        if effective_hop_seconds > self.window_seconds:
            raise AudioSegmentationError(
                "hop_seconds cannot exceed window_seconds; "
                "that would create gaps between segments."
            )

        if not (
            0.0
            < self.minimum_final_seconds
            <= self.window_seconds
        ):
            raise AudioSegmentationError(
                "minimum_final_seconds must be greater than zero "
                "and no larger than window_seconds."
            )

        if self.window_samples <= 0:
            raise AudioSegmentationError(
                "window_seconds must produce at least one sample."
            )

        if self.hop_samples <= 0:
            raise AudioSegmentationError(
                "hop_seconds must produce at least one sample."
            )

        if self.minimum_final_samples <= 0:
            raise AudioSegmentationError(
                "minimum_final_seconds must produce at least one sample."
            )

    @property
    def effective_hop_seconds(self) -> float:
        """
        Return the effective hop duration.

        Returns
        -------
        float
            Hop duration in seconds.
        """

        return (
            self.hop_seconds
            if self.hop_seconds is not None
            else self.window_seconds
        )

    @property
    def window_samples(self) -> int:
        """
        Return the fixed output window length in samples.

        Returns
        -------
        int
            Number of samples in each full segment.
        """

        return int(
            round(
                self.window_seconds
                * self.sample_rate
            )
        )

    @property
    def hop_samples(self) -> int:
        """
        Return the segment-start spacing in samples.

        Returns
        -------
        int
            Number of samples between consecutive segment starts.
        """

        return int(
            round(
                self.effective_hop_seconds
                * self.sample_rate
            )
        )

    @property
    def minimum_final_samples(self) -> int:
        """
        Return the minimum trailing segment size in samples.

        Returns
        -------
        int
            Minimum number of original samples required to retain
            a trailing partial segment.
        """

        return int(
            round(
                self.minimum_final_seconds
                * self.sample_rate
            )
        )


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """
    One fixed-window segment from a source waveform.

    Attributes
    ----------
    index:
        Zero-based segment index within the source recording.

    start_sample:
        Inclusive starting sample in the original waveform.

    end_sample:
        Exclusive ending sample in the original waveform.

        Important:
        This represents the real source audio boundary and does not
        include any padding.

    waveform:
        Float32 mono waveform.

        When padding is enabled, this always has exactly
        window_samples samples.

    is_padded:
        True when zero-padding was added to the segment.
    """

    index: int
    start_sample: int
    end_sample: int
    waveform: NDArray[np.float32]
    is_padded: bool

    @property
    def source_sample_count(self) -> int:
        """
        Return the number of real source samples.

        Returns
        -------
        int
            Number of samples originating from the source recording.
        """

        return self.end_sample - self.start_sample


class AudioSegmenter:
    """
    Segment mono waveforms using deterministic fixed-size windows.
    """

    def __init__(
        self,
        config: SegmentationConfig,
    ) -> None:
        """
        Create a segmenter.

        Parameters
        ----------
        config:
            Validated segmentation configuration.
        """

        self._config = config

    @property
    def config(self) -> SegmentationConfig:
        """
        Return the immutable segmentation configuration.
        """

        return self._config

    def segment(
        self,
        waveform: NDArray[np.floating],
    ) -> tuple[AudioSegment, ...]:
        """
        Split a mono waveform into deterministic fixed-window segments.

        Parameters
        ----------
        waveform:
            One-dimensional finite mono audio waveform.

        Returns
        -------
        tuple[AudioSegment, ...]
            Ordered collection of generated segments.

        Raises
        ------
        AudioSegmentationError
            If the waveform is not one-dimensional, numeric,
            or contains NaN/infinite values.
        """

        self._validate_waveform(waveform)

        mono_waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if mono_waveform.size == 0:
            return ()

        segments: list[AudioSegment] = []

        window_samples = self._config.window_samples
        hop_samples = self._config.hop_samples
        minimum_final_samples = (
            self._config.minimum_final_samples
        )

        last_covered_end_sample = 0

        for start_sample in range(
            0,
            mono_waveform.size,
            hop_samples,
        ):
            end_sample = min(
                start_sample + window_samples,
                mono_waveform.size,
            )

            source_waveform = mono_waveform[
                start_sample:end_sample
            ]

            source_sample_count = source_waveform.size

            is_final_partial = (
                source_sample_count < window_samples
            )

            # ------------------------------------------------------
            # Prevent redundant trailing overlapping segments.
            # ------------------------------------------------------
            #
            # With overlapping windows, the final partial window can
            # sometimes contain only audio already covered by the
            # previous full window.
            #
            # Example:
            #
            # window = 4 sec
            # hop    = 2 sec
            #
            # A final partial window starting at 8 sec may end at
            # 10 sec while the previous window already covered
            # 6-10 sec.
            #
            # Such a segment adds no new source audio.
            # ------------------------------------------------------

            if (
                is_final_partial
                and end_sample <= last_covered_end_sample
            ):
                break

            # ------------------------------------------------------
            # Reject extremely short trailing audio.
            # ------------------------------------------------------

            if (
                is_final_partial
                and source_sample_count
                < minimum_final_samples
            ):
                break

            # ------------------------------------------------------
            # Handle padding.
            # ------------------------------------------------------

            is_padded = (
                is_final_partial
                and self._config.pad_final_segment
            )

            if is_padded:
                padding_samples = (
                    window_samples
                    - source_sample_count
                )

                segment_waveform = np.pad(
                    source_waveform,
                    (
                        0,
                        padding_samples,
                    ),
                    mode="constant",
                ).astype(
                    np.float32,
                    copy=False,
                )

            else:
                segment_waveform = (
                    source_waveform.copy()
                )

            # ------------------------------------------------------
            # Create segment object.
            # ------------------------------------------------------

            segments.append(
                AudioSegment(
                    index=len(segments),
                    start_sample=start_sample,
                    end_sample=end_sample,
                    waveform=segment_waveform,
                    is_padded=is_padded,
                )
            )

            last_covered_end_sample = max(
                last_covered_end_sample,
                end_sample,
            )

            # Once the final partial segment has been emitted,
            # segmentation is complete.
            if is_final_partial:
                break

        return tuple(segments)

    @staticmethod
    def _validate_waveform(
        waveform: NDArray[np.floating],
    ) -> None:
        """
        Validate a waveform before segmentation.

        Parameters
        ----------
        waveform:
            Candidate mono waveform.

        Raises
        ------
        AudioSegmentationError
            If waveform dimensionality, datatype, or values are invalid.
        """

        waveform_array = np.asarray(waveform)

        if waveform_array.ndim != 1:
            raise AudioSegmentationError(
                "Expected a one-dimensional mono waveform, "
                f"got {waveform_array.ndim}D."
            )

        if not np.issubdtype(
            waveform_array.dtype,
            np.number,
        ):
            raise AudioSegmentationError(
                "Waveform samples must be numeric."
            )

        if not np.all(
            np.isfinite(waveform_array)
        ):
            raise AudioSegmentationError(
                "Waveform contains NaN or infinite values."
            )