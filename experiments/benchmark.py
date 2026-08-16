"""Flat lm-eval benchmark runner for one pruning configuration.

This module intentionally delegates scoring to lm_eval.simple_evaluate. It only
loads/provenances the model, removes requested layers, wraps the model in HFLM,
and writes append-only run/sample records.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))


MODEL_REVISIONS = {
    "base": {
        "model_id": "Qwen/Qwen2.5-7B",
        "revision": "d149729398750b98c0af14eb82c78cfe92750796",
    },
    "instruct": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    },
    "math": {
        "model_id": "Qwen/Qwen2.5-Math-7B-Instruct",
        "revision": "ef9926d75ab1d54532f6a30dd5e760355eb9aa4d",
    },
}

HARNESS_REVISION = "8a07e1110d060de48cfc7a9a7987b7659060b60b"
EVALUATION_SEED = 1234
EVALUATION_MAX_LENGTH = 2048
GPU_MEMORY_FRACTION = 0.90

TASKS = [
    "arc_challenge",
    "piqa",
    "winogrande",
    "hellaswag",
    "lambada_openai",
    "wikitext",
]

PRUNE_K = [2, 4, 8]
RANDOM_SEEDS = [0, 1, 2]
STRATEGIES = ["baseline", "bi", "random"]


@dataclass(frozen=True)
class RunConfig:
    model_key: str
    strategy: str
    k: int
    seed: int | None
    official_run: bool = False
    output_dir: Path = Path("results/lm_eval")
    batch_size: str = "4"
    device: str | None = None
    dtype: str = "auto"
    limit: int | float | None = None
    calibration_path: Path | None = None
    bi_scores_path: Path | None = None

    @property
    def run_key(self) -> str:
        seed = "none" if self.seed is None else str(self.seed)
        return f"{self.model_key}:{self.strategy}:k{self.k}:seed{seed}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_global_seeds(seed: int) -> None:
    torch = import_module("torch")
    numpy = import_module("numpy")
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def get_git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def enforce_clean_worktree() -> None:
    dirty = get_git_dirty()
    if dirty:
        raise RuntimeError("Official runs require a clean git worktree.")
    if dirty is None:
        raise RuntimeError("Official runs require git worktree status, but git status failed.")
    if get_git_sha() is None:
        raise RuntimeError("Official runs require an exact code git SHA, but git rev-parse failed.")


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def installed_harness_revision() -> str | None:
    try:
        distribution = importlib.metadata.distribution("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return None
    metadata = json.loads(direct_url)
    vcs_revision = metadata.get("vcs_info", {}).get("commit_id")
    if vcs_revision is not None:
        return vcs_revision

    source_url = metadata.get("url")
    if isinstance(source_url, str) and source_url.startswith("file:"):
        parsed = urlparse(source_url)
        source_path = Path(unquote(parsed.path.lstrip("/")))
        try:
            return subprocess.check_output(
                ["git", "-C", str(source_path), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return None


def enforce_harness_revision() -> None:
    actual = installed_harness_revision()
    if actual != HARNESS_REVISION:
        raise RuntimeError(
            f"Installed lm-evaluation-harness revision {actual!r} does not match "
            f"pinned {HARNESS_REVISION}."
        )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_revision_info(model_key: str) -> dict[str, str]:
    if model_key not in MODEL_REVISIONS:
        raise ValueError(f"Unknown model key {model_key!r}. Expected one of {sorted(MODEL_REVISIONS)}.")
    return MODEL_REVISIONS[model_key]


def torch_dtype_from_name(name: str, device: str) -> Any:
    torch = import_module("torch")
    if name == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}.")
    return mapping[name]


def load_model_and_tokenizer(model_key: str, device: str, dtype: str) -> tuple[Any, Any, dict[str, Any]]:
    transformers = import_module("transformers")
    info = model_revision_info(model_key)
    torch_dtype = torch_dtype_from_name(dtype, device)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        info["model_id"],
        revision=info["revision"],
        dtype=torch_dtype,
        trust_remote_code=False,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        info["model_id"],
        revision=info["revision"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    actual_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if actual_revision != info["revision"]:
        raise RuntimeError(
            f"Loaded model revision {actual_revision} does not match pinned {info['revision']}."
        )

    first_parameter = next(model.parameters())
    provenance = {
        "model_id": info["model_id"],
        "expected_revision": info["revision"],
        "loaded_revision": actual_revision,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "torch_dtype": str(torch_dtype),
        "parameter_dtype": str(first_parameter.dtype),
        "parameter_device": str(first_parameter.device),
        "use_cache": getattr(model.config, "use_cache", None),
    }
    return model, tokenizer, provenance


def discover_layer_pool(model: Any) -> tuple[Any, str, list[int]]:
    from src.handlers import UniversalHandler

    handler = UniversalHandler(model, verbose=False)
    component = "main" if "main" in handler.list_components() else handler.list_components()[0]
    layers = list(range(len(handler.get_layers(component))))
    return handler, component, layers


def load_calibration_records(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict) or "text" not in parsed or "text_sha256" not in parsed:
            raise ValueError(f"Calibration row {line_number} lacks text or text_sha256.")
        text = parsed["text"]
        if not isinstance(text, str):
            raise TypeError(f"Calibration row {line_number} text must be a string.")
        actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual_hash != parsed["text_sha256"]:
            raise RuntimeError(f"Calibration row {line_number} text hash does not match.")
        records.append(text)
    if not records:
        raise ValueError("Calibration file contains no text records.")
    return records


def calibration_batches(
    texts: list[str], tokenizer: Any, *, mode: str
) -> Iterable[dict[str, Any]]:
    if mode == "canonical":
        batch_size = 2
        padding = "longest"
        max_length = 256
    elif mode == "legacy":
        batch_size = 8
        padding = "max_length"
        max_length = 128
    else:
        raise ValueError(f"Unsupported BI mode {mode!r}.")

    for start in range(0, len(texts), batch_size):
        yield tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=padding,
            truncation=True,
            max_length=max_length,
        )


def compute_bi_bundle(
    model: Any,
    layer_modules: Iterable[Any],
    tokenizer: Any,
    calibration_path: Path | None,
) -> dict[str, Any]:
    if calibration_path is None:
        raise ValueError("BI computation requires --calibration-jsonl.")
    from src.core.block_influence import compute_block_influence
    from scipy.stats import spearmanr

    texts = load_calibration_records(calibration_path)
    modules = list(layer_modules)
    canonical = compute_block_influence(
        model,
        modules,
        calibration_batches(texts, tokenizer, mode="canonical"),
        mode="canonical",
    )
    legacy = compute_block_influence(
        model,
        modules,
        calibration_batches(texts, tokenizer, mode="legacy"),
        mode="legacy",
    )
    correlation_value = float(spearmanr(
        [canonical[idx] for idx in range(len(modules))],
        [legacy[idx] for idx in range(len(modules))],
    ).statistic)
    correlation = correlation_value if math.isfinite(correlation_value) else None
    return {
        "schema_version": 1,
        "model_name_or_path": getattr(model.config, "_name_or_path", None),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "calibration_path": str(calibration_path),
        "calibration_sha256": file_sha256(calibration_path),
        "calibration_examples": len(texts),
        "definitions": {
            "canonical": {
                "cosine": "per_token_hidden_dimension_fp32",
                "padding": "longest_per_batch",
                "attention_mask": True,
                "max_length": 256,
                "batch_size": 2,
                "aggregation": "token_mean_then_equal_example_mean",
            },
            "legacy": {
                "cosine": "flattened_sequence_and_hidden_fp16",
                "padding": "max_length",
                "attention_mask": False,
                "max_length": 128,
                "batch_size": 8,
                "aggregation": "equal_example_mean",
            },
        },
        "canonical": {str(idx): score for idx, score in canonical.items()},
        "legacy": {str(idx): score for idx, score in legacy.items()},
        "rank_correlation_spearman": correlation,
        "rank_correlation_defined": correlation is not None,
    }


def load_or_compute_bi_bundle(
    path: Path | None,
    model: Any,
    layer_modules: Iterable[Any],
    tokenizer: Any,
    calibration_path: Path | None,
) -> dict[str, Any] | None:
    modules = list(layer_modules)
    if path is not None and path.exists():
        bundle = json.loads(path.read_text(encoding="utf-8"))
        actual_revision = getattr(model.config, "_commit_hash", None)
        if bundle.get("model_revision") != actual_revision:
            raise RuntimeError(
                f"BI bundle revision {bundle.get('model_revision')!r} does not match "
                f"loaded model revision {actual_revision!r}."
            )
        if calibration_path is not None:
            actual_calibration_sha = file_sha256(calibration_path)
            if bundle.get("calibration_sha256") != actual_calibration_sha:
                raise RuntimeError("BI bundle calibration hash does not match --calibration-jsonl.")
        expected_indices = {str(idx) for idx in range(len(modules))}
        for mode in ("canonical", "legacy"):
            if set(bundle.get(mode, {})) != expected_indices:
                raise RuntimeError(
                    f"BI bundle {mode} vector does not match {len(modules)} model layers."
                )
        return bundle
    if calibration_path is None:
        return None

    bundle = compute_bi_bundle(model, modules, tokenizer, calibration_path)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return bundle


def removed_indices_for_config(
    strategy: str,
    k: int,
    seed: int | None,
    all_layers: list[int],
    bi_scores: dict[int, float] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    if strategy == "baseline":
        if k != 0:
            raise ValueError("Baseline runs must use k=0.")
        return [], {"all_layer_pool": all_layers, "permutation": [], "bi_scores": None}

    if k <= 0:
        raise ValueError("Pruned runs require k > 0.")
    if k > len(all_layers):
        raise ValueError(f"Cannot remove {k} layers from {len(all_layers)} discovered layers.")

    if strategy == "random":
        if seed is None:
            raise ValueError("Random pruning requires an explicit seed.")
        rng = random.Random(seed)
        permutation = list(all_layers)
        rng.shuffle(permutation)
        return sorted(permutation[:k]), {
            "all_layer_pool": all_layers,
            "permutation": permutation,
            "bi_scores": None,
        }

    if strategy == "bi":
        if bi_scores is None:
            raise ValueError("BI pruning requires a complete BI score vector.")
        missing = [idx for idx in all_layers if idx not in bi_scores]
        if missing:
            raise ValueError(f"BI score vector is missing layer indices: {missing}")
        ordered = sorted(all_layers, key=lambda idx: (bi_scores[idx], idx))
        return sorted(ordered[:k]), {
            "all_layer_pool": all_layers,
            "permutation": ordered,
            "bi_scores": {str(idx): bi_scores[idx] for idx in all_layers},
        }

    raise ValueError(f"Unsupported strategy {strategy!r}.")


def make_hflm(model: Any, tokenizer: Any, batch_size: str) -> Any:
    from lm_eval.models.huggingface import HFLM

    if batch_size.startswith("auto"):
        raise ValueError(
            "batch_size=auto is disabled: lm-eval batch probing exhausted host memory "
            "on the target laptop. Pass an explicit positive integer."
        )
    try:
        fixed_batch_size = int(batch_size)
    except ValueError as exc:
        raise ValueError("batch_size must be an explicit positive integer.") from exc
    if fixed_batch_size <= 0:
        raise ValueError("batch_size must be an explicit positive integer.")
    return HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=fixed_batch_size,
        max_length=EVALUATION_MAX_LENGTH,
    )


def simple_evaluate_one_task(hflm: Any, task: str, limit: int | float | None) -> tuple[dict[str, Any], float]:
    from lm_eval import simple_evaluate

    started = time.perf_counter()
    kwargs = {
        "model": hflm,
        "tasks": [task],
        "log_samples": True,
        "num_fewshot": 0,
        "apply_chat_template": False,
        "random_seed": EVALUATION_SEED,
        "numpy_random_seed": EVALUATION_SEED,
        "torch_random_seed": EVALUATION_SEED,
        "fewshot_random_seed": EVALUATION_SEED,
    }
    if limit is not None:
        kwargs["limit"] = limit
    result = simple_evaluate(**kwargs)
    return result, time.perf_counter() - started


def merge_task_results(task_results: list[tuple[str, dict[str, Any], float]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    merged: dict[str, Any] = {"results": {}, "configs": {}, "versions": {}, "n-shot": {}, "samples": {}}
    samples: list[dict[str, Any]] = []

    for task, result, elapsed in task_results:
        for key in ("results", "configs", "versions", "n-shot"):
            if isinstance(result.get(key), dict):
                merged[key].update(result[key])
        if "config" in result and "config" not in merged:
            merged["config"] = result["config"]
        merged.setdefault("task_timings_seconds", {})[task] = elapsed

        raw_samples = result.get("samples") or result.get("logs") or {}
        task_samples = raw_samples.get(task, raw_samples) if isinstance(raw_samples, dict) else raw_samples
        if isinstance(task_samples, list):
            for idx, sample in enumerate(task_samples):
                samples.append({"task": task, "sample_index": idx, "sample": sample})

    return merged, samples


def append_unique_jsonl(path: Path, record: dict[str, Any], unique_key: str = "run_id") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            existing = json.loads(line)
            if existing.get(unique_key) == record.get(unique_key):
                raise RuntimeError(f"Duplicate {unique_key} {record.get(unique_key)!r} in {path}.")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def append_sample_jsonl(path: Path, run_id: str, samples: list[dict[str, Any]]) -> None:
    if path.exists():
        raise RuntimeError(f"Sample log already exists: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            record = {"run_id": run_id, **sample}
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def start_gpu_monitor(path: Path) -> tuple[subprocess.Popen[str], Any, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    error_path = path.with_suffix(".stderr.log")
    output_handle = path.open("w", encoding="utf-8")
    error_handle = error_path.open("w", encoding="utf-8")
    query = (
        "timestamp,temperature.gpu,memory.used,clocks.current.sm,"
        "clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,"
        "clocks_event_reasons.hw_thermal_slowdown,"
        "clocks_event_reasons.sw_thermal_slowdown"
    )
    process = subprocess.Popen(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            "--loop-ms=5000",
        ],
        stdout=output_handle,
        stderr=error_handle,
        text=True,
    )
    return process, output_handle, error_handle


def stop_gpu_monitor(
    monitor: tuple[subprocess.Popen[str], Any, Any], path: Path
) -> dict[str, Any]:
    process, output_handle, error_handle = monitor
    process.terminate()
    process.wait(timeout=15)
    output_handle.close()
    error_handle.close()

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            raise RuntimeError(f"Unexpected nvidia-smi telemetry row: {line!r}")
        rows.append(fields)
    if not rows:
        raise RuntimeError("nvidia-smi produced no GPU telemetry rows.")

    temperatures = [int(row[1]) for row in rows]
    memory_mib = [int(row[2]) for row in rows]
    clocks_mhz = [int(row[3]) for row in rows]
    flags = {
        "software_power_cap": any(row[4] == "Active" for row in rows),
        "hardware_slowdown": any(row[5] == "Active" for row in rows),
        "hardware_thermal_slowdown": any(row[6] == "Active" for row in rows),
        "software_thermal_slowdown": any(row[7] == "Active" for row in rows),
    }
    return {
        "sample_count": len(rows),
        "sample_interval_seconds": 5,
        "max_temperature_c": max(temperatures),
        "max_memory_used_mib_nvidia_smi": max(memory_mib),
        "min_sm_clock_mhz": min(clocks_mhz),
        "max_sm_clock_mhz": max(clocks_mhz),
        "throttling_observed": flags,
        "telemetry_path": str(path),
    }


def base_provenance(config: RunConfig, device: str) -> dict[str, Any]:
    torch = import_module("torch")
    gpu = None
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        gpu = {
            "name": torch.cuda.get_device_name(idx),
            "device_index": idx,
            "capability": torch.cuda.get_device_capability(idx),
        }
    return {
        "created_at": utc_now(),
        "run_key": config.run_key,
        "code_sha": get_git_sha(),
        "worktree_dirty": get_git_dirty(),
        "benchmark_file_sha256": file_sha256(Path(__file__)),
        "harness": "lm_eval",
        "harness_version": package_version("lm_eval"),
        "harness_expected_sha": HARNESS_REVISION,
        "harness_installed_sha": installed_harness_revision(),
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "lm_eval": package_version("lm_eval"),
            "datasets": package_version("datasets"),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "matplotlib": package_version("matplotlib"),
        },
        "device": device,
        "gpu": gpu,
        "seeds": {"evaluation": EVALUATION_SEED, "strategy": config.seed},
        "tasks": TASKS,
        "limit": config.limit,
        "batch_size": config.batch_size,
        "max_length_tokens": EVALUATION_MAX_LENGTH,
        "gpu_memory_fraction": GPU_MEMORY_FRACTION if device == "cuda" else None,
    }


def run_configuration(config: RunConfig, run_id: str | None = None) -> dict[str, Any]:
    if config.official_run:
        enforce_clean_worktree()
        enforce_harness_revision()

    if config.strategy not in STRATEGIES:
        raise ValueError(f"Strategy must be one of {STRATEGIES}.")
    if config.model_key not in MODEL_REVISIONS:
        raise ValueError(f"Model key must be one of {sorted(MODEL_REVISIONS)}.")

    set_global_seeds(EVALUATION_SEED)

    torch = import_module("torch")
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_id = run_id or uuid.uuid4().hex
    started_wall = utc_now()
    started = time.perf_counter()
    run_record: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at": started_wall,
        "config": {
            "model_key": config.model_key,
            "strategy": config.strategy,
            "k": config.k,
            "seed": config.seed,
            "official_run": config.official_run,
        },
        "provenance": base_provenance(config, device),
    }

    runs_path = config.output_dir / "runs.jsonl"
    file_key = config.run_key.replace(":", "_")
    samples_path = config.output_dir / "samples" / f"{file_key}__{run_id}.jsonl"
    telemetry_path = config.output_dir / "telemetry" / f"{file_key}__{run_id}.csv"
    monitor = None

    try:
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
            torch.cuda.reset_peak_memory_stats()
            monitor = start_gpu_monitor(telemetry_path)
        model_load_started = time.perf_counter()
        model, tokenizer, model_provenance = load_model_and_tokenizer(config.model_key, device, config.dtype)
        model_load_seconds = time.perf_counter() - model_load_started
        run_record["provenance"]["model"] = model_provenance
        handler, component, all_layers = discover_layer_pool(model)

        bi_started = time.perf_counter()
        bi_bundle = load_or_compute_bi_bundle(
            config.bi_scores_path,
            model,
            handler.get_layers(component),
            tokenizer,
            config.calibration_path,
        )
        bi_seconds = time.perf_counter() - bi_started
        if config.official_run and bi_bundle is None:
            raise RuntimeError(
                "Official runs require --bi-scores or --calibration-jsonl so every "
                "record contains canonical and legacy BI vectors."
            )
        bi_scores = None
        if bi_bundle is not None:
            bi_scores = {int(idx): float(score) for idx, score in bi_bundle["canonical"].items()}

        removed, selection = removed_indices_for_config(
            config.strategy, config.k, config.seed, all_layers, bi_scores
        )
        if removed:
            handler.remove_layers(component, removed, inplace=True)

        hflm = make_hflm(model, tokenizer, config.batch_size)
        task_results = []
        for task in TASKS:
            result, elapsed = simple_evaluate_one_task(hflm, task, config.limit)
            task_results.append((task, result, elapsed))
        merged, samples = merge_task_results(task_results)
        gpu_runtime = None
        if monitor is not None:
            gpu_runtime = stop_gpu_monitor(monitor, telemetry_path)
            monitor = None
            gpu_runtime.update(
                {
                    "peak_allocated_mib_torch": torch.cuda.max_memory_allocated() / (1024**2),
                    "peak_reserved_mib_torch": torch.cuda.max_memory_reserved() / (1024**2),
                }
            )

        append_sample_jsonl(samples_path, run_id, samples)
        run_record.update(
            {
                "status": "succeeded",
                "finished_at": utc_now(),
                "timing_seconds": {
                    "total": time.perf_counter() - started,
                    "model_load": model_load_seconds,
                    "block_influence": bi_seconds,
                    "evaluation": sum(merged.get("task_timings_seconds", {}).values()),
                    "tasks": merged.get("task_timings_seconds", {}),
                },
                "pruning": {
                    "component": component,
                    "k": config.k,
                    "strategy": config.strategy,
                    "removed_indices": removed,
                    **selection,
                },
                "block_influence": bi_bundle,
                "results": merged,
                "sample_log_path": str(samples_path),
                "gpu_runtime": gpu_runtime,
            }
        )
        append_unique_jsonl(runs_path, run_record)
        return run_record
    except Exception as exc:
        if monitor is not None:
            try:
                run_record["gpu_runtime"] = stop_gpu_monitor(monitor, telemetry_path)
            except (OSError, RuntimeError, subprocess.SubprocessError) as monitor_exc:
                run_record["gpu_monitor_error"] = {
                    "type": type(monitor_exc).__name__,
                    "message": str(monitor_exc),
                }
        run_record.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "timing_seconds": {"total": time.perf_counter() - started},
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        append_unique_jsonl(runs_path, run_record)
        raise


def build_manifest() -> dict[str, Any]:
    configs = []
    for model_key, model in MODEL_REVISIONS.items():
        configs.append(
            {
                "run_key": f"{model_key}:baseline:k0:seednone",
                "model_key": model_key,
                "model_id": model["model_id"],
                "revision": model["revision"],
                "strategy": "baseline",
                "k": 0,
                "seed": None,
            }
        )
        for k in PRUNE_K:
            configs.append(
                {
                    "run_key": f"{model_key}:bi:k{k}:seednone",
                    "model_key": model_key,
                    "model_id": model["model_id"],
                    "revision": model["revision"],
                    "strategy": "bi",
                    "k": k,
                    "seed": None,
                }
            )
            for seed in RANDOM_SEEDS:
                configs.append(
                    {
                        "run_key": f"{model_key}:random:k{k}:seed{seed}",
                        "model_key": model_key,
                        "model_id": model["model_id"],
                        "revision": model["revision"],
                        "strategy": "random",
                        "k": k,
                        "seed": seed,
                    }
                )
    return {
        "schema_version": 1,
        "created_for": "flat_lm_eval_pruning_experiment",
        "harness_revision": HARNESS_REVISION,
        "evaluation_seed": EVALUATION_SEED,
        "models": MODEL_REVISIONS,
        "tasks": TASKS,
        "evaluation": {
            "dtype": "float16",
            "batch_size": 4,
            "max_length_tokens": EVALUATION_MAX_LENGTH,
            "gpu_memory_fraction": GPU_MEMORY_FRACTION,
            "num_fewshot": 0,
            "log_samples": True,
            "full_task_sets": True,
        },
        "grid": {
            "k": PRUNE_K,
            "random_seeds": RANDOM_SEEDS,
            "strategies": ["baseline", "bi", "random"],
            "config_count": len(configs),
            "notes": "Per model: baseline, deterministic BI at k=2/4/8, and random at k=2/4/8 for seeds 0/1/2.",
        },
        "configs": configs,
    }


def write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(build_manifest(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one lm-eval pruning configuration.")
    parser.add_argument("--model-key", choices=sorted(MODEL_REVISIONS), required=False)
    parser.add_argument("--strategy", choices=STRATEGIES, default="baseline")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/lm_eval"))
    parser.add_argument(
        "--batch-size",
        default="4",
        help="Fixed positive integer. Auto probing is disabled because it can exhaust host RAM.",
    )
    parser.add_argument("--device")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--limit", type=float)
    parser.add_argument("--official-run", action="store_true")
    parser.add_argument("--calibration-jsonl", type=Path)
    parser.add_argument("--bi-scores", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.write_manifest:
        write_manifest(args.write_manifest)
        return 0
    if args.model_key is None:
        raise SystemExit("--model-key is required unless --write-manifest is used.")

    limit: int | float | None = None
    if args.limit is not None:
        limit = int(args.limit) if args.limit >= 1 else args.limit

    config = RunConfig(
        model_key=args.model_key,
        strategy=args.strategy,
        k=args.k,
        seed=args.seed,
        official_run=args.official_run,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        limit=limit,
        calibration_path=args.calibration_jsonl,
        bi_scores_path=args.bi_scores,
    )
    record = run_configuration(config)
    print(json.dumps({"run_id": record["run_id"], "status": record["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
