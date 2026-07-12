"""
Audio utility functions.
"""

import soundfile as sf


def get_audio_metadata(file_path):
    """
    Read metadata from an audio file.
    """

    try:

        info = sf.info(file_path)

        duration = round(info.frames / info.samplerate, 2)

        return {
            "duration_sec": duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "status": "OK",
        }

    except Exception:

        return {
            "duration_sec": None,
            "sample_rate": None,
            "channels": None,
            "subtype": None,
            "status": "CORRUPTED",
        }