import json
import hashlib
from pathlib import Path

import pytest

import experiments.run_grid as run_grid


def write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def load_manifest_and_protocol():
    manifest_path = Path("experiments/experiment_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = run_grid.load_protocol(manifest)
    return manifest_path, manifest, protocol


def successful_record(config: dict, *, removed_indices: list[int] | None = None) -> dict:
    removed = [] if removed_indices is None else removed_indices
    selection_source = config.get("selection_source")
    pruning = {
        "strategy": config["strategy"],
        "k": config["k"],
        "removed_indices": removed,
    }
    if selection_source is not None:
        pruning["selection_source"] = selection_source
    return {
        "status": "succeeded",
        "config": {
            "official_run": True,
            "model_key": config["model_key"],
            "strategy": config["strategy"],
            "k": config["k"],
            "seed": config["seed"],
            "tasks": config["tasks"],
        },
        "provenance": {
            "run_key": config["run_key"],
            "limit": None,
            "batch_size": "4",
            "device": "cuda",
            "worktree_dirty": False,
            "tasks": config["tasks"],
            "seeds": {
                "evaluation": 1234,
                "strategy": config["seed"] if config["strategy"] == "random" else None,
            },
            "harness_expected_sha": "8a07e1110d060de48cfc7a9a7987b7659060b60b",
            "harness_installed_sha": "8a07e1110d060de48cfc7a9a7987b7659060b60b",
            "model": {
                "model_id": config["model_id"],
                "expected_revision": config["revision"],
                "loaded_revision": config["revision"],
                "parameter_dtype": "torch.float16",
            },
        },
        "pruning": pruning,
        "block_influence": {
            "canonical": {str(idx): float(idx) for idx in range(28)},
        },
    }


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


def test_successful_run_keys_rejects_records_that_do_not_match_manifest(tmp_path):
    runs = tmp_path / "runs.jsonl"
    manifest_config = {
        "run_key": "base:random:k4:seed3",
        "model_key": "base",
        "strategy": "random",
        "k": 4,
        "seed": 3,
        "tasks": ["wikitext"],
        "removed_indices": [1, 2, 3, 4],
        "selection_source": "conditional_bi_label_permutation",
    }
    write_records(
        runs,
        [
            {
                "status": "succeeded",
                "config": {
                    "official_run": True,
                    "model_key": "base",
                    "strategy": "random",
                    "k": 4,
                    "seed": 3,
                    "tasks": ["wikitext"],
                },
                "provenance": {"run_key": "base:random:k4:seed3", "limit": 10},
                "pruning": {
                    "strategy": "random",
                    "k": 4,
                    "removed_indices": [1, 2, 3, 4],
                    "selection_source": "conditional_bi_label_permutation",
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="limit"):
        run_grid.successful_run_keys([runs], [manifest_config])


def test_successful_run_keys_rejects_duplicate_success_records(tmp_path):
    runs = tmp_path / "runs.jsonl"
    manifest_config = {
        "run_key": "base:baseline:k0:seednone",
        "model_key": "base",
        "strategy": "baseline",
        "k": 0,
        "seed": None,
        "tasks": ["wikitext"],
    }
    record = {
        "status": "succeeded",
        "config": {
            "official_run": True,
            "model_key": "base",
            "strategy": "baseline",
            "k": 0,
            "seed": None,
            "tasks": ["wikitext"],
        },
        "provenance": {
            "run_key": "base:baseline:k0:seednone",
            "limit": None,
            "batch_size": "4",
            "device": "cuda",
            "worktree_dirty": False,
            "tasks": ["wikitext"],
        },
        "pruning": {"strategy": "baseline", "k": 0, "removed_indices": []},
    }
    write_records(runs, [record, record])

    with pytest.raises(ValueError, match="Duplicate succeeded official record"):
        run_grid.successful_run_keys([runs], [manifest_config])


def test_successful_run_keys_requires_manifest_hash_for_modern_records(tmp_path):
    manifest_path, manifest, protocol = load_manifest_and_protocol()
    config = next(
        entry for entry in manifest["configs"]
        if entry["run_key"] == "base:random:k4:seed3"
    )
    runs = tmp_path / "runs.jsonl"
    write_records(runs, [successful_record(config, removed_indices=config["removed_indices"])])

    with pytest.raises(ValueError, match="missing protocol manifest hash"):
        run_grid.successful_run_keys(
            [runs],
            manifest["configs"],
            manifest_path,
            manifest,
            protocol,
        )


def test_successful_run_keys_uses_protocol_not_candidate_bi_scores(tmp_path):
    manifest_path, manifest, protocol = load_manifest_and_protocol()
    config = next(
        entry for entry in manifest["configs"]
        if entry["run_key"] == "base:bi:k4:seednone"
    )
    record = successful_record(config, removed_indices=[0, 1, 2, 3])
    record["provenance"]["protocol_manifest"] = {
        "path": str(manifest_path),
        "sha256": run_grid.file_sha256(manifest_path),
    }
    record["block_influence"]["canonical"] = {
        str(idx): (0.0 if idx in {0, 1, 2, 3} else 1.0)
        for idx in range(28)
    }
    runs = tmp_path / "runs.jsonl"
    write_records(runs, [record])

    with pytest.raises(ValueError, match="removed indices"):
        run_grid.successful_run_keys(
            [runs],
            manifest["configs"],
            manifest_path,
            manifest,
            protocol,
        )


@pytest.mark.parametrize(
    "mutate,pattern",
    [
        (lambda record: record["provenance"].__setitem__("device", "cpu"), "CUDA"),
        (lambda record: record["provenance"].__setitem__("batch_size", "1"), "batch size"),
        (
            lambda record: record["provenance"]["model"].__setitem__(
                "loaded_revision", "wrong"
            ),
            "loaded model revision",
        ),
        (
            lambda record: record["provenance"]["model"].__setitem__(
                "parameter_dtype", "torch.float32"
            ),
            "model dtype",
        ),
    ],
)
def test_successful_run_keys_rejects_bad_runtime_provenance(tmp_path, mutate, pattern):
    manifest_path, manifest, protocol = load_manifest_and_protocol()
    config = next(
        entry for entry in manifest["configs"]
        if entry["run_key"] == "base:random:k4:seed3"
    )
    record = successful_record(config, removed_indices=config["removed_indices"])
    record["provenance"]["protocol_manifest"] = {
        "path": str(manifest_path),
        "sha256": run_grid.file_sha256(manifest_path),
    }
    mutate(record)
    runs = tmp_path / "runs.jsonl"
    write_records(runs, [record])

    with pytest.raises(ValueError, match=pattern):
        run_grid.successful_run_keys(
            [runs],
            manifest["configs"],
            manifest_path,
            manifest,
            protocol,
        )


def test_successful_run_keys_accepts_manifest_line_ending_hash_variant(tmp_path):
    manifest_path, manifest, protocol = load_manifest_and_protocol()
    variants = run_grid.text_file_sha256_variants(manifest_path)
    alternate_sha = next(
        (sha for sha in variants if sha != run_grid.file_sha256(manifest_path)),
        None,
    )
    if alternate_sha is None:
        pytest.skip("Manifest has no distinct LF/CRLF digest variant.")
    config = next(
        entry for entry in manifest["configs"]
        if entry["run_key"] == "base:random:k4:seed3"
    )
    record = successful_record(config, removed_indices=config["removed_indices"])
    record["provenance"]["protocol_manifest"] = {
        "path": str(manifest_path),
        "sha256": alternate_sha,
    }
    runs = tmp_path / "runs.jsonl"
    write_records(runs, [record])

    assert run_grid.successful_run_keys(
        [runs],
        manifest["configs"],
        manifest_path,
        manifest,
        protocol,
    ) == {"base:random:k4:seed3"}


def test_load_protocol_accepts_bi_line_ending_hash_variant(tmp_path):
    bi_path = tmp_path / "bi.json"
    lf_payload = b'{\n  "canonical": {}\n}\n'
    bi_path.write_bytes(lf_payload)
    crlf_sha = hashlib.sha256(lf_payload.replace(b"\n", b"\r\n")).hexdigest()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "models": {
                    "base": {
                        "bi_path": str(bi_path),
                        "bi_sha256": crlf_sha,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = run_grid.load_protocol({"protocol_path": str(protocol_path)})

    assert loaded["models"]["base"]["bi_sha256"] == crlf_sha


def test_benchmark_command_records_random_seed_and_model_bi_path(tmp_path, monkeypatch):
    monkeypatch.setattr(run_grid.sys, "executable", "python")
    config = {
        "model_key": "math",
        "strategy": "random",
        "k": 4,
        "seed": 2,
        "tasks": ["wikitext"],
        "removed_indices": [1, 4, 8, 12],
        "selection_source": "conditional_bi_label_permutation",
    }

    command = run_grid.benchmark_command(
        config,
        tmp_path / "manifest.json",
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
    assert command[command.index("--protocol-manifest") + 1] == str(tmp_path / "manifest.json")
    assert command[command.index("--tasks") + 1] == "wikitext"
    removed_position = command.index("--removed-indices")
    assert command[removed_position + 1 : removed_position + 5] == ["1", "4", "8", "12"]
    assert command[command.index("--selection-source") + 1] == "conditional_bi_label_permutation"


def test_load_manifest_rejects_duplicate_run_keys(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "grid": {"config_count": 2},
                "configs": [
                    {"run_key": "duplicate", "tasks": ["wikitext"]},
                    {"run_key": "duplicate", "tasks": ["wikitext"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        run_grid.load_manifest(manifest)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("Expected duplicate run_key validation to fail.")
