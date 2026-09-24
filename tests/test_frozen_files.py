import hashlib

import pytest

from experiments.frozen_files import require_frozen_text_hash, text_file_sha256_variants


def test_frozen_text_accepts_only_checkout_newline_conversion(tmp_path):
    historical = b'{\r\n  "score": 0.12\r\n}\r\n'
    expected = hashlib.sha256(historical).hexdigest()
    path = tmp_path / "scores.json"
    path.write_bytes(historical.replace(b"\r\n", b"\n"))
    require_frozen_text_hash(path, expected)

    path.write_bytes(historical.replace(b"0.12", b"0.13"))
    with pytest.raises(ValueError, match="Frozen text hash mismatch"):
        require_frozen_text_hash(path, expected)

    path.write_bytes(b'{"score":0.12}\n')
    with pytest.raises(ValueError, match="Frozen text hash mismatch"):
        require_frozen_text_hash(path, expected)


def test_frozen_text_does_not_ignore_missing_final_newline(tmp_path):
    path = tmp_path / "scores.json"
    path.write_bytes(b'{}\n')
    assert hashlib.sha256(b'{}').hexdigest() not in text_file_sha256_variants(path)
