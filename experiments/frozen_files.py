"""Match frozen text-file digests across Git's LF/CRLF checkout conversion.

Historical digests came from a Windows checkout. Only line endings may vary;
JSON values, spacing, encoding, and final-newline presence remain significant.
Never use this normalization for raw inference logs or binary artifacts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def text_file_sha256_variants(path: Path) -> set[str]:
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return {
        hashlib.sha256(payload).hexdigest()
        for payload in (raw, lf, lf.replace(b"\n", b"\r\n"))
    }


def require_frozen_text_hash(path: Path, expected: str) -> None:
    if expected not in text_file_sha256_variants(path):
        raise ValueError(f"Frozen text hash mismatch: {path}")
