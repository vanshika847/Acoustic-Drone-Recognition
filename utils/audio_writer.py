"""
Audio writing utilities.
"""

from pathlib import Path

import soundfile as sf


def save_audio(audio, sample_rate, output_path):
    """
    Save processed audio.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        file=str(output_path),
        data=audio,
        samplerate=sample_rate,
        subtype="PCM_16"
    )