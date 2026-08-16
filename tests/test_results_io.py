import json

import pytest

from experiments.benchmark import append_sample_jsonl, append_unique_jsonl


def test_append_unique_jsonl_rejects_duplicate_run_id(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_unique_jsonl(path, {"run_id": "abc", "status": "failed"})

    with pytest.raises(RuntimeError, match="Duplicate run_id"):
        append_unique_jsonl(path, {"run_id": "abc", "status": "succeeded"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "failed"


def test_append_sample_jsonl_writes_one_record_per_sample(tmp_path):
    path = tmp_path / "samples.jsonl"

    append_sample_jsonl(
        path,
        "run-1",
        [{"task": "piqa", "sample_index": 0, "sample": {"answer": "A"}}],
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["run_id"] == "run-1"
    assert record["task"] == "piqa"
    assert record["sample"]["answer"] == "A"

    with pytest.raises(RuntimeError, match="already exists"):
        append_sample_jsonl(path, "run-1", [])


def test_jsonl_writer_fails_on_unserializable_values(tmp_path):
    with pytest.raises(TypeError):
        append_unique_jsonl(tmp_path / "runs.jsonl", {"run_id": "abc", "bad": object()})
