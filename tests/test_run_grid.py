import json
from pathlib import Path

import experiments.run_grid as run_grid


def write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_successful_run_keys_only_accepts_official_successes(tmp_path):
    runs = tmp_path / "runs.jsonl"
    write_records(
        runs,
        [
            {
                "status": "succeeded",
                "config": {"official_run": True},
                "provenance": {"run_key": "base:baseline:k0:seednone"},
            },
            {
                "status": "failed",
                "config": {"official_run": True},
                "provenance": {"run_key": "base:bi:k2:seednone"},
            },
            {
                "status": "succeeded",
                "config": {"official_run": False},
                "provenance": {"run_key": "base:random:k2:seed0"},
            },
        ],
    )

    assert run_grid.successful_run_keys([runs]) == {"base:baseline:k0:seednone"}


def test_benchmark_command_records_random_seed_and_model_bi_path(tmp_path, monkeypatch):
    monkeypatch.setattr(run_grid.sys, "executable", "python")
    config = {"model_key": "math", "strategy": "random", "k": 4, "seed": 2}

    command = run_grid.benchmark_command(
        config,
        tmp_path / "runs",
        tmp_path / "calibration.jsonl",
        tmp_path / "bi",
        "4",
        "cuda",
        "float16",
    )

    assert command[0] == "python"
    assert command[command.index("--seed") + 1] == "2"
    assert command[command.index("--bi-scores") + 1] == str(tmp_path / "bi" / "math.json")
    assert "--official-run" in command


def test_load_manifest_rejects_duplicate_run_keys(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"configs": [{"run_key": "duplicate"}, {"run_key": "duplicate"}]}),
        encoding="utf-8",
    )

    try:
        run_grid.load_manifest(manifest)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("Expected duplicate run_key validation to fail.")
