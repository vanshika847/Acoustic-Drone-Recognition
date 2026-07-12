"""Reusable fixed-window segmentation for mono audio waveforms.

Purpose
-------
Convert a variable-length mono waveform into fixed-length clips suitable for
machine-learning feature extraction and batch training.  The module is pure:
it does not read files, write files, or alter labels.

Inputs
------
A one-dimensional finite NumPy waveform and a :class:`SegmentationConfig`.

Outputs
-------
An ordered sequence of :class:`AudioSegment` objects, each carrying its audio
samples, original timing bounds, and whether zero-padding was applied.

Dependencies
------------
NumPy only.

Algorithm
---------
Starting at sample zero, a fixed-size window advances by ``hop_samples``. Full
windows are emitted directly.  A final partial window is emitted only if it is
at least the configured minimum duration, and is zero-padded when requested.
Runtime is ``O(number_of_output_samples)`` and memory is ``O(number_of_output
samples)`` for the returned clips.

Example
-------
>>> import numpy as np
>>> config = SegmentationConfig(sample_rate=16_000, window_seconds=4.0)
>>> segments = AudioSegmenter(config).segment(np.zeros(64_000, dtype=np.float32))
>>> len(segments)
1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class AudioSegmentationError(ValueError):
    """Raised when audio or segmentation settings are invalid."""


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    """Parameters governing deterministic waveform segmentation.

    Attributes:
        sample_rate: Waveform sample rate in Hertz.
        window_seconds: Target duration of each emitted segment.
        hop_seconds: Offset between adjacent segment starts. Defaults to the
            window duration, producing non-overlapping segments.
        minimum_final_seconds: Minimum duration required to keep a trailing
            partial segment before optional padding.
        pad_final_segment: Whether retained trailing segments are zero-padded
            to the full window length.
    """

    sample_rate: int
    window_seconds: float
    hop_seconds: float | None = None
    minimum_final_seconds: float = 1.0
    pad_final_segment: bool = True

    def __post_init__(self) -> None:
        """Validate duration and sample-rate constraints at construction time."""

        effective_hop_seconds = self.hop_seconds or self.window_seconds
        if self.sample_rate <= 0:
            raise AudioSegmentationError("sample_rate must be greater than zero.")
        if self.window_seconds <= 0.0:
            raise AudioSegmentationError("window_seconds must be greater than zero.")
        if effective_hop_seconds <= 0.0:
            raise AudioSegmentationError("hop_seconds must be greater than zero.")
        if effective_hop_seconds > self.window_seconds:
            raise AudioSegmentationError(
                "hop_seconds cannot exceed window_seconds; that would discard audio."
            )
        if not 0.0 < self.minimum_final_seconds <= self.window_seconds:
            raise AudioSegmentationError(
                "minimum_final_seconds must be greater than zero and no larger "
                "than window_seconds."
            )
        if self.window_samples <= 0 or self.hop_samples <= 0:
            raise AudioSegmentationError(
                "window_seconds and hop_seconds must produce at least one sample."
            )
        if self.minimum_final_samples <= 0:
            raise AudioSegmentationError(
                "minimum_final_seconds must produce at least one sample."
            )

    @property
    def effective_hop_seconds(self) -> float:
        """Return the configured hop or the non-overlapping default hop.

        Returns:
            Effective segment-start spacing in seconds.
        """

        return self.hop_seconds or self.window_seconds

    @property
    def window_samples(self) -> int:
        """Return the fixed output window length in samples.

        Returns:
            Positive integer number of samples in an emitted full window.
        """

        return int(round(self.window_seconds * self.sample_rate))

    @property
    def hop_samples(self) -> int:
        """Return the step between segment starts in samples.

        Returns:
            Positive integer sample offset.
        """

        return int(round(self.effective_hop_seconds * self.sample_rate))

    @property
    def minimum_final_samples(self) -> int:
        """Return the shortest trailing segment eligible for retention.

        Returns:
            Positive integer sample count.
        """

        return int(round(self.minimum_final_seconds * self.sample_rate))


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """One fixed-window segment and its position in the source waveform.

    Attributes:
        index: Zero-based segment index within the source waveform.
        start_sample: Inclusive source sample index.
        end_sample: Exclusive unpadded source sample index.
        waveform: Float32 mono waveform; fixed-size when padding is enabled.
        is_padded: Whether trailing zeros were added to complete the window.
    """

    index: int
    start_sample: int
    end_sample: int
    waveform: NDArray[np.float32]
    is_padded: bool

    @property
    def source_sample_count(self) -> int:
        """Return the number of non-padding source samples in this segment.

        Returns:
            Number of samples taken from the original waveform.
        """

        return self.end_sample - self.start_sample


class AudioSegmenter:
    """Segment mono waveforms according to one immutable configuration."""

    def __init__(self, config: SegmentationConfig) -> None:
        """Create a segmenter.

        Args:
            config: Validated segmentation parameters.
        """

        self._config = config

    @property
    def config(self) -> SegmentationConfig:
        """Return the immutable configuration used by this segmenter.

        Returns:
            Segmentation settings for this instance.
        """

        return self._config

    def segment(self, waveform: NDArray[np.floating]) -> tuple[AudioSegment, ...]:
        """Split a mono waveform into deterministic fixed-window segments.

        Args:
            waveform: One-dimensional, finite, mono audio samples.

        Returns:
            Ordered immutable sequence of accepted waveform segments.

        Raises:
            AudioSegmentationError: If waveform dimensionality or values are
                invalid.
        """

        self._validate_waveform(waveform)
        mono_waveform = np.asarray(waveform, dtype=np.float32)
        if mono_waveform.size == 0:
            return ()

        segments: list[AudioSegment] = []
        window_samples = self._config.window_samples
        last_covered_end_sample = 0
        for start_sample in range(0, mono_waveform.size, self._config.hop_samples):
            end_sample = min(start_sample + window_samples, mono_waveform.size)
            source_waveform = mono_waveform[start_sample:end_sample]
            is_final_partial = source_waveform.size < window_samples

            # With overlap, a trailing partial window may be fully contained in
            # the preceding full window. It contributes no new audio and would
            # bias the training set toward recording endings.
            if is_final_partial and end_sample <= last_covered_end_sample:
                break
            if is_final_partial and source_waveform.size < self._config.minimum_final_samples:
                break

            is_padded = is_final_partial and self._config.pad_final_segment
            if is_padded:
                segment_waveform = np.pad(
                    source_waveform,
                    (0, window_samples - source_waveform.size),
                    mode="constant",
                ).astype(np.float32, copy=False)
            else:
                segment_waveform = source_waveform.copy()

            segments.append(
                AudioSegment(
                    index=len(segments),
                    start_sample=start_sample,
                    end_sample=end_sample,
                    waveform=segment_waveform,
                    is_padded=is_padded,
                )
            )
            last_covered_end_sample = max(last_covered_end_sample, end_sample)

            if is_final_partial:
                break
        return tuple(segments)

    @staticmethod
    def _validate_waveform(waveform: NDArray[np.floating]) -> None:
        """Validate a waveform before segmentation.

        Args:
            waveform: Candidate audio waveform.

        Raises:
            AudioSegmentationError: If the waveform is not finite, numeric, or
                one-dimensional.
        """

        waveform_array = np.asarray(waveform)
        if waveform_array.ndim != 1:
            raise AudioSegmentationError(
                f"Expected a one-dimensional mono waveform, got {waveform_array.ndim}D."
            )
        if not np.issubdtype(waveform_array.dtype, np.number):
            raise AudioSegmentationError("Waveform samples must be numeric.")
        if not np.all(np.isfinite(waveform_array)):
            raise AudioSegmentationError("Waveform contains NaN or infinite values.")
