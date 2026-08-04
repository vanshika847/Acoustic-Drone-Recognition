"""
Generate SHA-256 hashes for files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

HASH_CHUNK_SIZE_BYTES = 1024 * 1024


def sha256_file(file_path: str | Path) -> str:
    """
    Compute the SHA-256 hash of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    file_path = Path(file_path)

    digest = hashlib.sha256()

    with file_path.open("rb") as audio_file:
        for chunk in iter(
            lambda: audio_file.read(HASH_CHUNK_SIZE_BYTES),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()