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


def test_jsonl_writer_serializes_explicit_technical_types(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "runs.jsonl"

    append_unique_jsonl(path, {"run_id": "abc", "dtype": torch.float16})

    assert json.loads(path.read_text(encoding="utf-8"))["dtype"] == "torch.float16"


def test_jsonl_writer_serializes_callable_by_stable_qualified_name(tmp_path):
    path = tmp_path / "runs.jsonl"

    append_unique_jsonl(path, {"run_id": "abc", "callable": test_jsonl_writer_fails_on_unserializable_values})

    callable_record = json.loads(path.read_text(encoding="utf-8"))["callable"]
    assert callable_record == {
        "python_callable": (
            "test_results_io.test_jsonl_writer_fails_on_unserializable_values"
        )
    }


def test_jsonl_writer_removes_machine_cache_path_from_harness_callable(tmp_path):
    path = tmp_path / "runs.jsonl"

    def harness_callable():
        return None

    harness_callable.__module__ = (
        r"C:\cache\lm-evaluation-harness\revision\lm_eval\tasks\example\utils"
    )
    append_unique_jsonl(path, {"run_id": "abc", "callable": harness_callable})

    callable_record = json.loads(path.read_text(encoding="utf-8"))["callable"]
    assert callable_record == {
        "python_callable": (
            "lm_eval.tasks.example.utils."
            "test_jsonl_writer_removes_machine_cache_path_from_harness_callable."
            "<locals>.harness_callable"
        )
    }
