"""Run the pinned pruning grid sequentially and resume from successful JSONL records."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from experiments.frozen_files import require_frozen_text_hash, text_file_sha256_variants


REPO_ROOT = Path(__file__).resolve().parent.parent
PRE_MANIFEST_RUN_KEYS = {
    "base:baseline:k0:seednone",
    "base:bi:k2:seednone",
    "base:random:k2:seed0",
    "base:random:k2:seed1",
    "base:random:k2:seed2",
}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    configs = manifest.get("configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError(f"Manifest {path} has no non-empty configs list.")
    run_keys = [config.get("run_key") for config in configs]
    if any(not isinstance(run_key, str) for run_key in run_keys):
        raise ValueError(f"Manifest {path} contains a config without a string run_key.")
    if len(set(run_keys)) != len(run_keys):
        raise ValueError(f"Manifest {path} contains duplicate run_key values.")
    expected_count = manifest.get("grid", {}).get("config_count")
    if expected_count != len(configs):
        raise ValueError(
            f"Manifest {path} declares {expected_count} configs but contains {len(configs)}."
        )
    for config in configs:
        tasks = config.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"Manifest config {config['run_key']} has no task list.")
        removed = config.get("removed_indices")
        if removed is not None:
            k = config.get("k")
            if len(removed) != k or len(set(removed)) != k:
                raise ValueError(
                    f"Manifest config {config['run_key']} does not contain {k} distinct indices."
                )
    return configs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    return path if path.is_absolute() else REPO_ROOT / path


def load_protocol(manifest: dict[str, Any]) -> dict[str, Any]:
    protocol_path = manifest.get("protocol_path")
    if not isinstance(protocol_path, str):
        raise ValueError("Manifest does not declare protocol_path.")
    path = resolve_repo_path(protocol_path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    for model_key, info in protocol.get("models", {}).items():
        bi_path = info.get("bi_path")
        bi_sha = info.get("bi_sha256")
        if isinstance(bi_path, str) and isinstance(bi_sha, str):
            require_frozen_text_hash(resolve_repo_path(bi_path), bi_sha)
    return protocol


def expected_bi_removed_indices(
    config: dict[str, Any],
    record: dict[str, Any],
    protocol: dict[str, Any] | None,
) -> list[int]:
    if protocol is None:
        raise ValueError("BI official success validation requires the frozen protocol.")
    model_key = config["model_key"]
    k = str(config["k"])
    indices = protocol.get("models", {}).get(model_key, {}).get("bi_indices", {}).get(k)
    if not isinstance(indices, list) and config["run_key"] in PRE_MANIFEST_RUN_KEYS:
        scores = record.get("block_influence", {}).get("canonical")
        if not isinstance(scores, dict):
            raise ValueError("Pre-manifest BI official success lacks a canonical BI vector.")
        selected = sorted(scores, key=lambda index: (float(scores[index]), int(index)))[: int(k)]
        return sorted(int(index) for index in selected)
    if not isinstance(indices, list):
        raise ValueError(f"Frozen protocol lacks BI indices for {model_key} k={k}.")
    return list(indices)


def expected_removed_indices(
    config: dict[str, Any],
    record: dict[str, Any],
    protocol: dict[str, Any] | None,
) -> list[int]:
    if "removed_indices" in config:
        return list(config["removed_indices"])
    if config["strategy"] == "bi":
        return expected_bi_removed_indices(config, record, protocol)
    return []


def validate_success_record(
    record: dict[str, Any],
    expected: dict[str, Any],
    source: str,
    manifest: dict[str, Any] | None,
    manifest_sha256_variants: set[str] | None,
    protocol: dict[str, Any] | None,
) -> None:
    run_key = expected["run_key"]
    recorded_config = record.get("config", {})
    for field in ("model_key", "strategy", "k", "seed"):
        if recorded_config.get(field) != expected.get(field):
            raise ValueError(
                f"Official success {source} does not match manifest {run_key}.{field}: "
                f"{recorded_config.get(field)!r} != {expected.get(field)!r}."
            )
    if recorded_config.get("official_run") is not True:
        raise ValueError(f"Official success {source} is not marked official.")

    provenance = record.get("provenance", {})
    if provenance.get("run_key") != run_key:
        raise ValueError(f"Official success {source} has the wrong run_key.")
    if provenance.get("limit") is not None:
        raise ValueError(f"Official success {source} used limit={provenance.get('limit')!r}.")
    if provenance.get("batch_size") != "4":
        raise ValueError(f"Official success {source} did not use batch size 4.")
    if provenance.get("device") != "cuda":
        raise ValueError(f"Official success {source} did not use CUDA.")
    if provenance.get("worktree_dirty") is not False:
        raise ValueError(f"Official success {source} used a dirty worktree.")
    recorded_tasks = provenance.get("tasks", recorded_config.get("tasks"))
    if recorded_tasks != expected["tasks"]:
        raise ValueError(
            f"Official success {source} task list does not match manifest {run_key}."
        )
    if manifest is not None:
        expected_seed = expected["seed"] if expected["strategy"] == "random" else None
        if provenance.get("seeds") != {
            "evaluation": manifest["evaluation_seed"],
            "strategy": expected_seed,
        }:
            raise ValueError(f"Official success {source} seed provenance does not match manifest.")
        harness_revision = manifest["harness_revision"]
        if provenance.get("harness_expected_sha") != harness_revision:
            raise ValueError(f"Official success {source} expected harness SHA is wrong.")
        if provenance.get("harness_installed_sha") != harness_revision:
            raise ValueError(f"Official success {source} installed harness SHA is wrong.")
        model = provenance.get("model", {})
        if model.get("model_id") != expected["model_id"]:
            raise ValueError(f"Official success {source} model_id does not match manifest.")
        if model.get("expected_revision") != expected["revision"]:
            raise ValueError(f"Official success {source} expected model revision is wrong.")
        if model.get("loaded_revision") != expected["revision"]:
            raise ValueError(f"Official success {source} loaded model revision is wrong.")
        if model.get("parameter_dtype") != "torch.float16":
            raise ValueError(f"Official success {source} model dtype is not torch.float16.")

    recorded_manifest = provenance.get("protocol_manifest")
    if manifest_sha256_variants is not None:
        if recorded_manifest is None and run_key not in PRE_MANIFEST_RUN_KEYS:
            raise ValueError(f"Official success {source} is missing protocol manifest hash.")
        if (
            recorded_manifest is not None
            and recorded_manifest.get("sha256") not in manifest_sha256_variants
        ):
            raise ValueError(f"Official success {source} used a different manifest hash.")

    pruning = record.get("pruning", {})
    if pruning.get("strategy") != expected["strategy"] or pruning.get("k") != expected["k"]:
        raise ValueError(f"Official success {source} pruning metadata does not match manifest.")
    expected_removed = expected_removed_indices(expected, record, protocol)
    if pruning.get("removed_indices") != expected_removed:
        raise ValueError(f"Official success {source} removed indices do not match manifest.")
    recorded_selection_source = pruning.get("selection_source")
    if (
        "selection_source" in expected
        and recorded_selection_source != expected["selection_source"]
        and not (run_key in PRE_MANIFEST_RUN_KEYS and recorded_selection_source is None)
    ):
        raise ValueError(f"Official success {source} selection_source does not match manifest.")


def successful_run_keys(
    run_files: Iterable[Path],
    manifest_configs: Iterable[dict[str, Any]] | None = None,
    manifest_path: Path | None = None,
    manifest: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
) -> set[str]:
    expected_by_key = None
    manifest_sha256_variants = None
    if manifest_configs is not None:
        expected_by_key = {config["run_key"]: config for config in manifest_configs}
        if manifest_path is not None and manifest_path.is_file():
            manifest_sha256_variants = text_file_sha256_variants(manifest_path)

    completed: set[str] = set()
    for path in sorted(run_files):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("status") != "succeeded":
                    continue
                if not record.get("config", {}).get("official_run", False):
                    continue
                run_key = record.get("provenance", {}).get("run_key")
                if not isinstance(run_key, str):
                    raise ValueError(f"Missing provenance.run_key in {path}:{line_number}.")
                if run_key in completed:
                    raise ValueError(f"Duplicate succeeded official record for {run_key}.")
                if expected_by_key is not None:
                    if run_key not in expected_by_key:
                        raise ValueError(f"Official success {path}:{line_number} is not in manifest.")
                    validate_success_record(
                        record,
                        expected_by_key[run_key],
                        f"{path}:{line_number}",
                        manifest,
                        manifest_sha256_variants,
                        protocol,
                    )
                completed.add(run_key)
    return completed


def benchmark_command(
    config: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
    calibration_path: Path,
    bi_dir: Path,
    batch_size: str,
    device: str,
    dtype: str,
) -> list[str]:
    model_key = str(config["model_key"])
    command = [
        sys.executable,
        "-u",
        "-m",
        "experiments.benchmark",
        "--model-key",
        model_key,
        "--strategy",
        str(config["strategy"]),
        "--k",
        str(config["k"]),
        "--official-run",
        "--device",
        device,
        "--dtype",
        dtype,
        "--batch-size",
        batch_size,
        "--calibration-jsonl",
        str(calibration_path),
        "--bi-scores",
        str(bi_dir / f"{model_key}.json"),
        "--output-dir",
        str(output_dir),
        "--protocol-manifest",
        str(manifest_path),
    ]
    seed = config.get("seed")
    if seed is not None:
        command.extend(["--seed", str(seed)])
    tasks = config.get("tasks")
    if tasks:
        command.extend(["--tasks", *(str(task) for task in tasks)])
    removed_indices = config.get("removed_indices")
    if removed_indices is not None:
        command.extend(["--removed-indices", *(str(idx) for idx in removed_indices)])
        command.extend(["--selection-source", str(config["selection_source"])])
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume the frozen official experiment manifest.")
    parser.add_argument("--official-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("experiments/experiment_manifest.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results/lm_eval"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/lm_eval/grid"))
    parser.add_argument("--bi-dir", type=Path, default=Path("experiments/bi"))
    parser.add_argument(
        "--calibration-jsonl",
        type=Path,
        default=Path("experiments/calibration/wikitext_2_raw_v1_seed1234_n128.jsonl"),
    )
    parser.add_argument("--batch-size", default="4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.official_run:
        raise SystemExit("Refusing to run the grid without --official-run.")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    configs = load_manifest(args.manifest)
    protocol = load_protocol(manifest)
    completed = successful_run_keys(
        args.results_root.rglob("runs.jsonl"),
        configs,
        args.manifest,
        manifest,
        protocol,
    )
    pending = [config for config in configs if config["run_key"] not in completed]
    print(f"Official grid: {len(completed)} completed, {len(pending)} pending.", flush=True)

    for position, config in enumerate(pending, start=1):
        run_key = config["run_key"]
        command = benchmark_command(
            config,
            args.manifest,
            args.output_dir,
            args.calibration_jsonl,
            args.bi_dir,
            args.batch_size,
            args.device,
            args.dtype,
        )
        print(f"[{position}/{len(pending)}] {run_key}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    print("Official grid queue finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
