"""
Generate SHA256 hashes for files.
"""

import hashlib


def sha256_file(file_path):

    sha = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            block = f.read(8192)

            if not block:
                break

            sha.update(block)

    return sha.hexdigest()