"""Run the pinned pruning grid sequentially and resume from successful JSONL records."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent


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


def successful_run_keys(run_files: Iterable[Path]) -> set[str]:
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

    configs = load_manifest(args.manifest)
    completed = successful_run_keys(args.results_root.rglob("runs.jsonl"))
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
