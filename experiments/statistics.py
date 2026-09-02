"""Frozen statistical analysis for the 2026-08-18 permutation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest, rankdata

from experiments.benchmark import decode_non_finite_float, json_safe


MODEL_KEYS = ("base", "instruct", "math")
PERMUTATION_SEEDS = tuple(range(3, 23))
FULL_TASK_SEEDS = tuple(range(3, 8))
PRIMARY_K = 4
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_ITERATIONS = 10_000
MC_PRIMARY_METRICS = {
    "arc_challenge": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "hellaswag": "acc_norm",
    "lambada_openai": "acc",
}
EXPECTED_TASK_SAMPLE_COUNTS = {
    "arc_challenge": 1_172,
    "piqa": 1_838,
    "winogrande": 1_267,
    "hellaswag": 10_042,
    "lambada_openai": 5_153,
    "wikitext": 62,
}
PRE_MANIFEST_RUN_KEYS = {
    "base:baseline:k0:seednone",
    "base:bi:k2:seednone",
    "base:random:k2:seed0",
    "base:random:k2:seed1",
    "base:random:k2:seed2",
}
REPO_ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def public_path(path_value: str | Path) -> str:
    """Return a stable repo-relative path without publishing workstation directories."""

    path = resolve_repo_path(path_value).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def sample_identity(sample: dict[str, Any], task: str, path: Path) -> str:
    doc_id = sample.get("doc_id")
    doc_hash = sample.get("doc_hash")
    if doc_id is None or not isinstance(doc_hash, str) or not doc_hash:
        raise ValueError(
            f"Missing stable {task} sample identity in {path}: "
            f"doc_id={doc_id!r}, doc_hash={doc_hash!r}."
        )
    return json.dumps([doc_id, doc_hash], separators=(",", ":"))


def load_successful_official_records(results_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(results_root.rglob("runs.jsonl")):
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
                if run_key in records:
                    raise ValueError(f"Multiple successful official records exist for {run_key}.")
                records[run_key] = record
    return records


def require_records(
    records: dict[str, dict[str, Any]], run_keys: Iterable[str]
) -> list[dict[str, Any]]:
    run_keys = list(run_keys)
    missing = [run_key for run_key in run_keys if run_key not in records]
    if missing:
        raise ValueError(f"Missing successful official records: {missing}")
    return [records[run_key] for run_key in run_keys]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_official_records(
    records: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    protocol: dict[str, Any] | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    configs = manifest["configs"]
    expected = {config["run_key"]: config for config in configs}
    missing = sorted(set(expected) - set(records))
    extra = sorted(set(records) - set(expected))
    if missing or extra:
        raise ValueError(f"Official run-key mismatch: missing={missing}, extra={extra}.")

    inventory: dict[str, dict[str, Any]] = {}
    harness_revision = manifest["harness_revision"]
    for run_key in sorted(expected):
        config = expected[run_key]
        record = records[run_key]
        recorded_config = record["config"]
        for field in ("model_key", "strategy", "k", "seed"):
            if recorded_config.get(field) != config.get(field):
                raise ValueError(
                    f"Manifest mismatch for {run_key}.{field}: "
                    f"{recorded_config.get(field)!r} != {config.get(field)!r}."
                )
        if recorded_config.get("official_run") is not True:
            raise ValueError(f"Run {run_key} is not marked official.")

        provenance = record["provenance"]
        if provenance.get("tasks") != config["tasks"]:
            raise ValueError(f"Task list mismatch for {run_key}.")
        if provenance.get("limit") is not None:
            raise ValueError(f"Official run {run_key} used a dataset limit.")
        if provenance.get("batch_size") != "4":
            raise ValueError(f"Official run {run_key} did not use batch size 4.")
        if provenance.get("device") != "cuda":
            raise ValueError(f"Official run {run_key} did not use CUDA.")
        if provenance.get("worktree_dirty") is not False:
            raise ValueError(f"Official run {run_key} used a dirty worktree.")
        expected_strategy_seed = config["seed"] if config["strategy"] == "random" else None
        if provenance.get("seeds") != {
            "evaluation": manifest["evaluation_seed"],
            "strategy": expected_strategy_seed,
        }:
            raise ValueError(f"Seed provenance mismatch for {run_key}.")
        if provenance.get("harness_expected_sha") != harness_revision:
            raise ValueError(f"Expected harness revision mismatch for {run_key}.")
        if provenance.get("harness_installed_sha") != harness_revision:
            raise ValueError(f"Installed harness revision mismatch for {run_key}.")

        model = provenance["model"]
        for field in ("model_id", "expected_revision", "loaded_revision"):
            expected_value = config["model_id"] if field == "model_id" else config["revision"]
            if model.get(field) != expected_value:
                raise ValueError(f"Model provenance mismatch for {run_key}.{field}.")
        if model.get("parameter_dtype") != "torch.float16":
            raise ValueError(f"Model dtype mismatch for {run_key}.")

        recorded_manifest = provenance.get("protocol_manifest")
        if manifest_sha256 is not None:
            if recorded_manifest is None and run_key not in PRE_MANIFEST_RUN_KEYS:
                raise ValueError(f"Missing protocol-manifest provenance for {run_key}.")
            if (
                recorded_manifest is not None
                and recorded_manifest.get("sha256") != manifest_sha256
            ):
                raise ValueError(f"Protocol-manifest hash mismatch for {run_key}.")

        if "removed_indices" in config:
            expected_removed = config["removed_indices"]
        elif config["strategy"] == "bi":
            protocol_indices = (
                protocol.get("models", {})
                .get(config["model_key"], {})
                .get("bi_indices", {})
                .get(str(config["k"]))
                if protocol is not None
                else None
            )
            if protocol_indices is not None:
                expected_removed = protocol_indices
            elif run_key in PRE_MANIFEST_RUN_KEYS:
                scores = record["block_influence"]["canonical"]
                selected = sorted(
                    scores, key=lambda index: (float(scores[index]), int(index))
                )[: config["k"]]
                expected_removed = sorted(int(index) for index in selected)
            else:
                raise ValueError(f"Missing frozen BI selection for {run_key}.")
        else:
            expected_removed = []
        if record["pruning"].get("removed_indices") != expected_removed:
            raise ValueError(f"Removed-index mismatch for {run_key}.")
        recorded_selection_source = record["pruning"].get("selection_source")
        if "selection_source" in config:
            expected_selection_source = config["selection_source"]
            if recorded_selection_source is None and run_key in PRE_MANIFEST_RUN_KEYS:
                pass
            elif recorded_selection_source != expected_selection_source:
                raise ValueError(f"Selection-source mismatch for {run_key}.")

        sample_path = resolve_repo_path(record["sample_log_path"])
        if not sample_path.is_file():
            raise ValueError(f"Missing sample artifact for {run_key}: {sample_path}.")
        counts = {task: 0 for task in config["tasks"]}
        identities = {task: set() for task in config["tasks"]}
        wikitext_log_likelihoods: list[float] = []
        wikitext_word_counts: list[float] = []
        with sample_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if row.get("run_id") != record["run_id"]:
                    raise ValueError(
                        f"Sample run_id mismatch in {sample_path}:{line_number}."
                    )
                task = row.get("task")
                if task not in counts:
                    raise ValueError(
                        f"Unexpected task {task!r} in {sample_path}:{line_number}."
                    )
                identity = sample_identity(row["sample"], task, sample_path)
                if identity in identities[task]:
                    raise ValueError(
                        f"Duplicate {task} identity {identity} in {sample_path}."
                    )
                identities[task].add(identity)
                counts[task] += 1
                if task == "wikitext":
                    contribution = row["sample"].get("word_perplexity")
                    if not isinstance(contribution, list) or len(contribution) != 2:
                        raise ValueError(
                            f"Invalid WikiText contribution in {sample_path}:{line_number}."
                        )
                    log_likelihood = float(decode_non_finite_float(contribution[0]))
                    word_count = float(contribution[1])
                    if (
                        math.isnan(log_likelihood)
                        or log_likelihood == math.inf
                        or not math.isfinite(word_count)
                        or word_count <= 0
                    ):
                        raise ValueError(
                            f"Invalid WikiText contribution in {sample_path}:{line_number}."
                        )
                    wikitext_log_likelihoods.append(log_likelihood)
                    wikitext_word_counts.append(word_count)
        expected_counts = {
            task: EXPECTED_TASK_SAMPLE_COUNTS[task] for task in config["tasks"]
        }
        if counts != expected_counts:
            raise ValueError(
                f"Sample-count mismatch for {run_key}: {counts} != {expected_counts}."
            )

        reconstructed_wikitext_ppl = None
        if "wikitext" in counts:
            log_likelihood_sum = sum(wikitext_log_likelihoods)
            word_count_sum = sum(wikitext_word_counts)
            try:
                reconstructed_wikitext_ppl = math.exp(
                    -log_likelihood_sum / word_count_sum
                )
            except OverflowError:
                reconstructed_wikitext_ppl = math.inf
            recorded_wikitext_ppl = wikitext_word_perplexity(record)
            both_infinite = math.isinf(reconstructed_wikitext_ppl) and math.isinf(
                recorded_wikitext_ppl
            )
            if not both_infinite and not math.isclose(
                reconstructed_wikitext_ppl,
                recorded_wikitext_ppl,
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                raise ValueError(
                    f"WikiText aggregate/sample mismatch for {run_key}: "
                    f"{recorded_wikitext_ppl} != {reconstructed_wikitext_ppl}."
                )

        inventory[run_key] = {
            "run_id": record["run_id"],
            "record_sha256": canonical_json_sha256(record),
            "sample_log_path": public_path(sample_path),
            "sample_log_sha256": sha256_file(sample_path),
            "sample_counts": counts,
            "reconstructed_wikitext_word_perplexity": reconstructed_wikitext_ppl,
            "code_sha": provenance.get("code_sha"),
            "protocol_manifest_sha256": (
                recorded_manifest.get("sha256") if recorded_manifest is not None else None
            ),
            "selection_source_recorded": recorded_selection_source is not None,
        }
    return inventory


def distribution_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or np.any(np.isnan(array)):
        raise ValueError("Distribution summary requires non-empty non-NaN values.")
    standard_deviation = math.inf if np.any(np.isinf(array)) else float(array.std())
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": standard_deviation,
    }


def wikitext_word_perplexity(record: dict[str, Any]) -> float:
    raw_value = record["results"]["results"]["wikitext"]["word_perplexity,none"]
    value = float(decode_non_finite_float(raw_value))
    if math.isnan(value) or value == -math.inf or value <= 0:
        raise ValueError(f"Invalid WikiText word perplexity in {record['provenance']['run_key']}: {value}")
    return value


def exact_lower_is_better_p(observed: float, random_values: Iterable[float]) -> float:
    values = list(random_values)
    return (1 + sum(value <= observed for value in values)) / (len(values) + 1)


def protocol_selection(
    protocol: dict[str, Any], seed: int, model_key: str, k: int
) -> dict[str, Any]:
    matches = [entry for entry in protocol["permutations"] if entry["seed"] == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected one protocol permutation for seed {seed}, found {len(matches)}.")
    return matches[0]["by_model"][model_key][str(k)]


def primary_permutation_analysis(
    records: dict[str, dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    baselines = {
        model_key: wikitext_word_perplexity(
            require_records(records, [f"{model_key}:baseline:k0:seednone"])[0]
        )
        for model_key in MODEL_KEYS
    }
    bi_ppl = {
        model_key: wikitext_word_perplexity(
            require_records(records, [f"{model_key}:bi:k{PRIMARY_K}:seednone"])[0]
        )
        for model_key in MODEL_KEYS
    }
    random_ppl: dict[int, dict[str, float]] = {}
    for seed in PERMUTATION_SEEDS:
        random_ppl[seed] = {
            model_key: wikitext_word_perplexity(
                require_records(records, [f"{model_key}:random:k{PRIMARY_K}:seed{seed}"])[0]
            )
            for model_key in MODEL_KEYS
        }

    bi_log_degradation = {
        model_key: math.log(bi_ppl[model_key] / baselines[model_key])
        for model_key in MODEL_KEYS
    }
    random_log_degradation = {
        seed: {
            model_key: math.log(random_ppl[seed][model_key] / baselines[model_key])
            for model_key in MODEL_KEYS
        }
        for seed in PERMUTATION_SEEDS
    }
    bi_statistic = float(np.mean(list(bi_log_degradation.values())))
    random_statistics = {
        seed: float(np.mean(list(random_log_degradation[seed].values())))
        for seed in PERMUTATION_SEEDS
    }
    primary_p = exact_lower_is_better_p(bi_statistic, random_statistics.values())

    candidates = ["bi", *(str(seed) for seed in PERMUTATION_SEEDS)]
    rank_by_model: dict[str, dict[str, float]] = {}
    for model_key in MODEL_KEYS:
        values = [
            bi_log_degradation[model_key],
            *(random_log_degradation[seed][model_key] for seed in PERMUTATION_SEEDS),
        ]
        ranks = rankdata(values, method="average")
        rank_by_model[model_key] = {
            candidate: float(rank) for candidate, rank in zip(candidates, ranks, strict=True)
        }
    robust_bi = float(np.mean([rank_by_model[model_key]["bi"] for model_key in MODEL_KEYS]))
    robust_random = {
        seed: float(np.mean([rank_by_model[model_key][str(seed)] for model_key in MODEL_KEYS]))
        for seed in PERMUTATION_SEEDS
    }
    robust_p = exact_lower_is_better_p(robust_bi, robust_random.values())

    touches_edge = {
        seed: any(
            protocol_selection(protocol, seed, model_key, PRIMARY_K)["touches_edge"]
            for model_key in MODEL_KEYS
        )
        for seed in PERMUTATION_SEEDS
    }
    edge_free_seeds = [seed for seed in PERMUTATION_SEEDS if not touches_edge[seed]]
    edge_touching_seeds = [seed for seed in PERMUTATION_SEEDS if touches_edge[seed]]
    edge_layers = set(protocol["edge_layers"])
    bi_touches_edge = any(
        edge_layers.intersection(protocol["models"][model_key]["bi_indices"][str(PRIMARY_K)])
        for model_key in MODEL_KEYS
    )
    edge_free_p = None
    if not bi_touches_edge:
        edge_free_p = exact_lower_is_better_p(
            bi_statistic, (random_statistics[seed] for seed in edge_free_seeds)
        )

    model_specific = {}
    for model_key in MODEL_KEYS:
        random_values = [random_ppl[seed][model_key] for seed in PERMUTATION_SEEDS]
        random_log_values = [
            random_log_degradation[seed][model_key] for seed in PERMUTATION_SEEDS
        ]
        model_specific[model_key] = {
            "baseline_ppl": baselines[model_key],
            "bi_ppl": bi_ppl[model_key],
            "bi_to_baseline_ratio": bi_ppl[model_key] / baselines[model_key],
            "bi_log_degradation": bi_log_degradation[model_key],
            "bi_rank_lower_is_better": rank_by_model[model_key]["bi"],
            "bi_rank_denominator": len(PERMUTATION_SEEDS) + 1,
            "random_worse_than_bi_count": sum(
                value > bi_ppl[model_key] for value in random_values
            ),
            "descriptive_exact_p": exact_lower_is_better_p(
                bi_ppl[model_key], random_values
            ),
            "random_ppl": {
                str(seed): random_ppl[seed][model_key] for seed in PERMUTATION_SEEDS
            },
            "random_ppl_summary": distribution_summary(random_values),
            "random_log_degradation_summary": distribution_summary(random_log_values),
            "median_random_to_bi_ratio": float(np.median(random_values))
            / bi_ppl[model_key],
        }

    return {
        "primary": {
            "statistic": "mean_model_log_ppl_ratio_to_baseline",
            "bi": bi_statistic,
            "random": {
                str(seed): random_statistics[seed] for seed in PERMUTATION_SEEDS
            },
            "one_sided_exact_p": primary_p,
            "ties": "random_not_worse_counts_in_numerator",
        },
        "robustness": {
            "statistic": "mean_within_model_rank",
            "bi": robust_bi,
            "random": {str(seed): robust_random[seed] for seed in PERMUTATION_SEEDS},
            "one_sided_exact_p": robust_p,
        },
        "edge_diagnostic": {
            "edge_layers": protocol["edge_layers"],
            "bi_touches_edge": bi_touches_edge,
            "touches_edge_by_seed": {
                str(seed): touches_edge[seed] for seed in PERMUTATION_SEEDS
            },
            "edge_free_seeds": edge_free_seeds,
            "edge_free_count": len(edge_free_seeds),
            "edge_touching_seeds": edge_touching_seeds,
            "edge_touching_count": len(edge_touching_seeds),
            "bi_rank_p_among_edge_free": edge_free_p,
            "bi_log_degradation": bi_statistic,
            "edge_free_random_log_degradation": {
                str(seed): random_statistics[seed] for seed in edge_free_seeds
            },
            "edge_touching_random_log_degradation": {
                str(seed): random_statistics[seed] for seed in edge_touching_seeds
            },
            "edge_free_summary": (
                distribution_summary(
                    random_statistics[seed] for seed in edge_free_seeds
                )
                if edge_free_seeds
                else None
            ),
            "edge_touching_summary": (
                distribution_summary(
                    random_statistics[seed] for seed in edge_touching_seeds
                )
                if edge_touching_seeds
                else None
            ),
        },
        "model_specific_descriptive": model_specific,
    }


def aggregate_metric(record: dict[str, Any], task: str, metric: str) -> float:
    values = record["results"]["results"][task]
    candidates = [f"{metric},none", metric]
    keys = [key for key in candidates if key in values]
    if len(keys) != 1:
        raise ValueError(
            f"Expected one {task}.{metric} value in "
            f"{record['provenance']['run_key']}, found {keys}."
        )
    value = float(decode_non_finite_float(values[keys[0]]))
    if math.isnan(value):
        raise ValueError(
            f"NaN aggregate {task}.{metric} in {record['provenance']['run_key']}."
        )
    return value


def exploratory_k2_analysis(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_keys = {
        "baseline": "base:baseline:k0:seednone",
        "bi": "base:bi:k2:seednone",
        **{f"random_seed_{seed}": f"base:random:k2:seed{seed}" for seed in range(3)},
    }
    selected = {
        label: require_records(records, [run_key])[0] for label, run_key in run_keys.items()
    }
    all_metrics = {**MC_PRIMARY_METRICS, "wikitext": "word_perplexity"}
    baseline = selected["baseline"]
    baseline_metrics = {
        task: (
            wikitext_word_perplexity(baseline)
            if task == "wikitext"
            else aggregate_metric(baseline, task, metric)
        )
        for task, metric in all_metrics.items()
    }

    output: dict[str, Any] = {}
    for label, record in selected.items():
        metrics: dict[str, Any] = {}
        for task, metric in all_metrics.items():
            value = (
                wikitext_word_perplexity(record)
                if task == "wikitext"
                else aggregate_metric(record, task, metric)
            )
            comparison_key = (
                "ratio_to_baseline" if task == "wikitext" else "difference_from_baseline"
            )
            comparison = (
                value / baseline_metrics[task]
                if task == "wikitext"
                else value - baseline_metrics[task]
            )
            metrics[task] = {
                "metric": metric,
                "value": value,
                comparison_key: comparison,
            }
        output[label] = {
            "run_key": record["provenance"]["run_key"],
            "removed_indices": record["pruning"]["removed_indices"],
            "metrics": metrics,
        }
    return {
        "status": "exploratory_not_confirmatory",
        "model_key": "base",
        "candidates": output,
    }


def dose_response_analysis(
    records: dict[str, dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    baselines = {
        model_key: wikitext_word_perplexity(
            require_records(records, [f"{model_key}:baseline:k0:seednone"])[0]
        )
        for model_key in MODEL_KEYS
    }
    for k in (PRIMARY_K, int(protocol["secondary"]["dose_response_k"])):
        bi_ppl = {
            model_key: wikitext_word_perplexity(
                require_records(records, [f"{model_key}:bi:k{k}:seednone"])[0]
            )
            for model_key in MODEL_KEYS
        }
        random_ppl = {
            seed: {
                model_key: wikitext_word_perplexity(
                    require_records(records, [f"{model_key}:random:k{k}:seed{seed}"])[0]
                )
                for model_key in MODEL_KEYS
            }
            for seed in PERMUTATION_SEEDS
        }
        bi_log = {
            model_key: math.log(bi_ppl[model_key] / baselines[model_key])
            for model_key in MODEL_KEYS
        }
        random_log = {
            seed: {
                model_key: math.log(random_ppl[seed][model_key] / baselines[model_key])
                for model_key in MODEL_KEYS
            }
            for seed in PERMUTATION_SEEDS
        }
        bi_global = float(np.mean(list(bi_log.values())))
        random_global = {
            seed: float(np.mean(list(random_log[seed].values())))
            for seed in PERMUTATION_SEEDS
        }
        touches_edge = {
            seed: any(
                protocol_selection(protocol, seed, model_key, k)["touches_edge"]
                for model_key in MODEL_KEYS
            )
            for seed in PERMUTATION_SEEDS
        }

        model_specific: dict[str, Any] = {}
        for model_key in MODEL_KEYS:
            random_values = [random_ppl[seed][model_key] for seed in PERMUTATION_SEEDS]
            ranks = rankdata([bi_ppl[model_key], *random_values], method="average")
            model_specific[model_key] = {
                "baseline_ppl": baselines[model_key],
                "bi_ppl": bi_ppl[model_key],
                "bi_to_baseline_ratio": bi_ppl[model_key] / baselines[model_key],
                "bi_rank_lower_is_better": float(ranks[0]),
                "bi_rank_denominator": len(ranks),
                "random_worse_than_bi_count": sum(
                    value > bi_ppl[model_key] for value in random_values
                ),
                "random_ppl": {
                    str(seed): random_ppl[seed][model_key]
                    for seed in PERMUTATION_SEEDS
                },
                "random_ppl_summary": distribution_summary(random_values),
                "median_random_to_bi_ratio": float(np.median(random_values))
                / bi_ppl[model_key],
            }
        edge_free = [seed for seed in PERMUTATION_SEEDS if not touches_edge[seed]]
        edge_touching = [seed for seed in PERMUTATION_SEEDS if touches_edge[seed]]
        output[str(k)] = {
            "status": "primary" if k == PRIMARY_K else "secondary_descriptive",
            "global": {
                "statistic": "mean_model_log_ppl_ratio_to_baseline",
                "bi": bi_global,
                "bi_geometric_degradation_factor": math.exp(bi_global),
                "random": {
                    str(seed): random_global[seed] for seed in PERMUTATION_SEEDS
                },
                "random_summary": distribution_summary(random_global.values()),
                "random_geometric_degradation_factor_summary": distribution_summary(
                    math.exp(value) for value in random_global.values()
                ),
                "descriptive_exact_p": exact_lower_is_better_p(
                    bi_global, random_global.values()
                ),
            },
            "model_specific": model_specific,
            "edge_diagnostic": {
                "edge_free_seeds": edge_free,
                "edge_touching_seeds": edge_touching,
                "edge_free_random_summary": (
                    distribution_summary(random_global[seed] for seed in edge_free)
                    if edge_free
                    else None
                ),
                "edge_touching_random_summary": (
                    distribution_summary(random_global[seed] for seed in edge_touching)
                    if edge_touching
                    else None
                ),
            },
        }
    return output


def full_task_aggregate_summary(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model_key in MODEL_KEYS:
        baseline = require_records(records, [f"{model_key}:baseline:k0:seednone"])[0]
        bi = require_records(records, [f"{model_key}:bi:k{PRIMARY_K}:seednone"])[0]
        output[model_key] = {}
        for task, metric in MC_PRIMARY_METRICS.items():
            baseline_value = aggregate_metric(baseline, task, metric)
            bi_value = aggregate_metric(bi, task, metric)
            random = {
                str(seed): aggregate_metric(
                    require_records(
                        records, [f"{model_key}:random:k{PRIMARY_K}:seed{seed}"]
                    )[0],
                    task,
                    metric,
                )
                for seed in FULL_TASK_SEEDS
            }
            ranks = rankdata(
                [-bi_value, *(-value for value in random.values())], method="average"
            )
            random_deltas = [value - baseline_value for value in random.values()]
            output[model_key][task] = {
                "metric": metric,
                "baseline": baseline_value,
                "bi": bi_value,
                "bi_difference_from_baseline": bi_value - baseline_value,
                "bi_rank_higher_is_better": float(ranks[0]),
                "bi_rank_denominator": len(ranks),
                "random": random,
                "random_summary": distribution_summary(random.values()),
                "random_difference_from_baseline_summary": distribution_summary(
                    random_deltas
                ),
                "bi_minus_random_median": bi_value - float(np.median(list(random.values()))),
            }
    return output


def load_task_samples(record: dict[str, Any], task: str) -> dict[str, dict[str, Any]]:
    path = resolve_repo_path(record["sample_log_path"])
    samples: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["task"] != task:
                continue
            sample = row["sample"]
            key = sample_identity(sample, task, path)
            if key in samples:
                raise ValueError(f"Duplicate {task} sample key {key} in {path}.")
            samples[key] = sample
    if not samples:
        raise ValueError(f"No {task} samples found in {path}.")
    return samples


def wikitext_document_arrays(record: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    samples = load_task_samples(record, "wikitext")
    ordered_samples = sorted(samples.values(), key=lambda sample: sample["doc_hash"])
    sample_path = resolve_repo_path(record["sample_log_path"])
    keys = [
        sample_identity(sample, "wikitext", sample_path) for sample in ordered_samples
    ]
    log_likelihood = np.array(
        [
            float(decode_non_finite_float(sample["word_perplexity"][0]))
            for sample in ordered_samples
        ]
    )
    words = np.array(
        [float(sample["word_perplexity"][1]) for sample in ordered_samples]
    )
    if (
        np.any(words <= 0)
        or np.any(np.isnan(log_likelihood))
        or np.any(np.isposinf(log_likelihood))
    ):
        raise ValueError(f"Invalid WikiText document values in {record['provenance']['run_key']}.")
    return keys, log_likelihood, words


def weighted_log_likelihood_sum(counts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Sum document log likelihoods without producing 0 * -inf NaNs."""

    negative_infinity = np.isneginf(values)
    finite_values = np.where(negative_infinity, 0.0, values)
    totals = counts @ finite_values
    if np.any(negative_infinity):
        includes_catastrophic_document = counts[:, negative_infinity].sum(axis=1) > 0
        totals[includes_catastrophic_document] = -math.inf
    return totals


def paired_document_bootstrap(
    records: dict[str, dict[str, Any]], iterations: int = BOOTSTRAP_ITERATIONS
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    baseline_arrays: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    candidate_arrays: dict[str, dict[str, tuple[list[str], np.ndarray, np.ndarray]]] = {}
    candidate_names = ["bi", *(str(seed) for seed in PERMUTATION_SEEDS)]

    for model_key in MODEL_KEYS:
        baseline = require_records(records, [f"{model_key}:baseline:k0:seednone"])[0]
        baseline_arrays[model_key] = wikitext_document_arrays(baseline)
        candidate_arrays[model_key] = {}
        for candidate in candidate_names:
            strategy = "bi" if candidate == "bi" else "random"
            seed = "none" if candidate == "bi" else candidate
            record = require_records(
                records, [f"{model_key}:{strategy}:k{PRIMARY_K}:seed{seed}"]
            )[0]
            candidate_arrays[model_key][candidate] = wikitext_document_arrays(record)

    reference_keys = baseline_arrays[MODEL_KEYS[0]][0]
    for model_key in MODEL_KEYS:
        if baseline_arrays[model_key][0] != reference_keys:
            raise ValueError("WikiText baseline document keys are not aligned across models.")
        for candidate in candidate_names:
            if candidate_arrays[model_key][candidate][0] != reference_keys:
                raise ValueError(f"WikiText documents are not aligned for {model_key}/{candidate}.")

    document_count = len(reference_keys)
    observed_global = {candidate: 0.0 for candidate in candidate_names}
    for model_key in MODEL_KEYS:
        _, baseline_ll, baseline_words = baseline_arrays[model_key]
        if np.any(np.isneginf(baseline_ll)):
            raise ValueError(f"Baseline WikiText log likelihood is -inf for {model_key}.")
        for candidate in candidate_names:
            _, candidate_ll, candidate_words = candidate_arrays[model_key][candidate]
            if not np.array_equal(candidate_words, baseline_words):
                raise ValueError(f"WikiText word weights differ for {model_key}/{candidate}.")
            observed_global[candidate] += (
                (-candidate_ll.sum() + baseline_ll.sum()) / baseline_words.sum()
            ) / len(MODEL_KEYS)
    observed_advantage = float(
        np.median([observed_global[str(seed)] for seed in PERMUTATION_SEEDS])
        - observed_global["bi"]
    )

    counts = rng.multinomial(
        document_count,
        np.full(document_count, 1 / document_count),
        size=iterations,
    )
    global_values = {candidate: np.zeros(iterations) for candidate in candidate_names}
    for model_key in MODEL_KEYS:
        _, baseline_ll, baseline_words = baseline_arrays[model_key]
        baseline_ll_sum = weighted_log_likelihood_sum(counts, baseline_ll)
        baseline_word_sum = counts @ baseline_words
        for candidate in candidate_names:
            _, candidate_ll, candidate_words = candidate_arrays[model_key][candidate]
            if not np.array_equal(candidate_words, baseline_words):
                raise ValueError(f"WikiText word weights differ for {model_key}/{candidate}.")
            candidate_ll_sum = weighted_log_likelihood_sum(counts, candidate_ll)
            log_ratio = (-candidate_ll_sum + baseline_ll_sum) / baseline_word_sum
            global_values[candidate] += log_ratio / len(MODEL_KEYS)

    random_matrix = np.column_stack(
        [global_values[str(seed)] for seed in PERMUTATION_SEEDS]
    )
    advantage = np.median(random_matrix, axis=1) - global_values["bi"]
    lower, upper = np.quantile(advantage, [0.025, 0.975])
    return {
        "seed": BOOTSTRAP_SEED,
        "iterations": iterations,
        "document_count": document_count,
        "estimand": "median_random_global_log_degradation_minus_bi",
        "estimate": observed_advantage,
        "percentile_95_ci": [float(lower), float(upper)],
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: p_values[key])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[key])
        running_max = max(running_max, candidate)
        adjusted[key] = running_max
    return adjusted


def mcnemar_exact(bi: dict[str, dict[str, Any]], random_samples: dict[str, dict[str, Any]], metric: str) -> dict[str, Any]:
    if set(bi) != set(random_samples):
        raise ValueError("McNemar sample keys do not align.")
    bi_only = 0
    random_only = 0
    for key in bi:
        bi_correct = binary_correctness(bi[key][metric], f"bi/{key}/{metric}")
        random_correct = binary_correctness(
            random_samples[key][metric], f"random/{key}/{metric}"
        )
        bi_only += bi_correct and not random_correct
        random_only += random_correct and not bi_correct
    discordant = bi_only + random_only
    if discordant == 0:
        p_value = 1.0
        p_decimal = Decimal(1)
    else:
        numerator = sum(
            math.comb(discordant, successes)
            for successes in range(bi_only, discordant + 1)
        )
        denominator = 1 << discordant
        with localcontext() as context:
            context.prec = 50
            p_decimal = Decimal(numerator) / Decimal(denominator)
        p_value = float(p_decimal)
        scipy_value = float(
            binomtest(bi_only, discordant, p=0.5, alternative="greater").pvalue
        )
        if scipy_value != 0.0 and not math.isclose(
            scipy_value, p_value, rel_tol=1e-12, abs_tol=0.0
        ):
            raise RuntimeError(
                f"Exact McNemar probability disagrees with scipy: {p_value} != {scipy_value}."
            )
    return {
        "bi_only_correct": bi_only,
        "random_only_correct": random_only,
        "discordant": discordant,
        "one_sided_p": p_value,
        "one_sided_p_decimal": format(p_decimal, ".18E"),
    }


def binary_correctness(value: Any, context: str) -> bool:
    value = decode_non_finite_float(value)
    if isinstance(value, bool):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid binary metric at {context}: {value!r}") from exc
    if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
        raise ValueError(f"Invalid binary metric at {context}: {value!r}")
    return bool(numeric)


def full_task_mcnemar(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model_key in MODEL_KEYS:
        output[model_key] = {}
        bi_record = require_records(records, [f"{model_key}:bi:k{PRIMARY_K}:seednone"])[0]
        for task, metric in MC_PRIMARY_METRICS.items():
            bi_samples = load_task_samples(bi_record, task)
            raw: dict[str, dict[str, Any]] = {}
            p_values: dict[str, float] = {}
            decimal_p_values: dict[str, Decimal] = {}
            for seed in FULL_TASK_SEEDS:
                random_record = require_records(
                    records, [f"{model_key}:random:k{PRIMARY_K}:seed{seed}"]
                )[0]
                comparison = mcnemar_exact(
                    bi_samples, load_task_samples(random_record, task), metric
                )
                raw[str(seed)] = comparison
                p_values[str(seed)] = comparison["one_sided_p"]
                decimal_p_values[str(seed)] = Decimal(
                    comparison["one_sided_p_decimal"]
                )
            adjusted = holm_adjust(p_values)
            ordered = sorted(decimal_p_values, key=decimal_p_values.get)
            decimal_adjusted: dict[str, Decimal] = {}
            running_max = Decimal(0)
            for rank, seed_key in enumerate(ordered):
                candidate = min(
                    Decimal(1),
                    Decimal(len(ordered) - rank) * decimal_p_values[seed_key],
                )
                running_max = max(running_max, candidate)
                decimal_adjusted[seed_key] = running_max
            for seed in FULL_TASK_SEEDS:
                raw[str(seed)]["holm_adjusted_p"] = adjusted[str(seed)]
                raw[str(seed)]["holm_adjusted_p_decimal"] = format(
                    decimal_adjusted[str(seed)], ".18E"
                )
            output[model_key][task] = {"metric": metric, "comparisons": raw}
    return output


def binary_bootstrap_ci(values: Iterable[float], rng: np.random.Generator, iterations: int) -> dict[str, Any]:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0 or not np.all(np.isin(values, [0.0, 1.0])):
        raise ValueError("Binary bootstrap requires a non-empty vector containing only zero and one.")
    estimate = float(values.mean())
    replicates = rng.binomial(values.size, estimate, size=iterations) / values.size
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return {
        "estimate": estimate,
        "sample_count": int(values.size),
        "percentile_95_ci": [float(lower), float(upper)],
    }


def full_task_bootstrap_cis(
    records: dict[str, dict[str, Any]], iterations: int = BOOTSTRAP_ITERATIONS
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output: dict[str, Any] = {}
    for model_key in MODEL_KEYS:
        run_keys = [
            f"{model_key}:baseline:k0:seednone",
            f"{model_key}:bi:k{PRIMARY_K}:seednone",
            *(f"{model_key}:random:k{PRIMARY_K}:seed{seed}" for seed in FULL_TASK_SEEDS),
        ]
        output[model_key] = {}
        for record in require_records(records, run_keys):
            run_key = record["provenance"]["run_key"]
            output[model_key][run_key] = {}
            for task, metric in MC_PRIMARY_METRICS.items():
                samples = load_task_samples(record, task)
                output[model_key][run_key][task] = {
                    "metric": metric,
                    **binary_bootstrap_ci(
                        (sample[metric] for sample in samples.values()), rng, iterations
                    ),
                }
    return output


def git_state() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, bool(status.strip())


def audit_attempts(results_root: Path) -> dict[str, Any]:
    failed: list[dict[str, Any]] = []
    indexed_samples: set[Path] = set()
    for runs_path in sorted(results_root.rglob("runs.jsonl")):
        with runs_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_log = record.get("sample_log_path")
                if sample_log:
                    indexed_samples.add(resolve_repo_path(sample_log).resolve())
                if (
                    record.get("config", {}).get("official_run")
                    and record.get("status") == "failed"
                ):
                    failed.append(
                        {
                            "run_key": record.get("provenance", {}).get("run_key"),
                            "run_id": record.get("run_id"),
                            "error": record.get("error"),
                            "runs_path": public_path(runs_path),
                        }
                    )
    all_samples = {path.resolve() for path in results_root.rglob("samples/*.jsonl")}
    unindexed = sorted(public_path(path) for path in all_samples - indexed_samples)
    return {
        "failed_official_attempt_count": len(failed),
        "failed_official_attempts": failed,
        "unindexed_sample_file_count": len(unindexed),
        "unindexed_sample_files": unindexed,
    }


def analysis_provenance(
    args: argparse.Namespace,
    inventory: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    revision, tracked_dirty = git_state()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_git_sha": revision,
        "tracked_worktree_dirty": tracked_dirty,
        "analysis_file": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "analysis_file_sha256": sha256_file(Path(__file__)),
        "protocol": {
            "path": public_path(args.protocol),
            "sha256": sha256_file(resolve_repo_path(args.protocol)),
        },
        "manifest": {
            "path": public_path(args.manifest),
            "sha256": sha256_file(resolve_repo_path(args.manifest)),
        },
        "results_root": public_path(args.results_root),
        "input_run_count": len(inventory),
        "input_inventory_sha256": canonical_json_sha256(inventory),
        "input_code_shas": sorted(
            {entry["code_sha"] for entry in inventory.values() if entry["code_sha"]}
        ),
        "input_runs": inventory,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": args.bootstrap_iterations,
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "matplotlib": package_version("matplotlib"),
        },
    }


def write_official_record_export(
    path: Path,
    records: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    ordered = [records[config["run_key"]] for config in manifest["configs"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            public_record = json_safe(record)
            for task_config in public_record.get("results", {}).get("configs", {}).values():
                metadata = task_config.get("metadata", {})
                config_source = metadata.get("config_source")
                if not isinstance(config_source, str):
                    continue
                parts = PureWindowsPath(config_source).parts
                if "lm_eval" in parts:
                    metadata["config_source"] = "/".join(parts[parts.index("lm_eval") :])
                elif PureWindowsPath(config_source).is_absolute():
                    metadata["config_source"] = f"<external>/{PureWindowsPath(config_source).name}"
            handle.write(
                json.dumps(
                    public_record,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen pruning analysis.")
    parser.add_argument("--results-root", type=Path, default=Path("results/lm_eval"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/permutation_protocol.json")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("experiments/experiment_manifest.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/confirmatory/summary.json")
    )
    parser.add_argument(
        "--records-output",
        type=Path,
        default=Path("results/confirmatory/official_runs.jsonl"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = read_json(resolve_repo_path(args.protocol))
    manifest = read_json(resolve_repo_path(args.manifest))
    results_root = resolve_repo_path(args.results_root)
    records = load_successful_official_records(results_root)
    inventory = validate_official_records(
        records,
        manifest,
        protocol=protocol,
        manifest_sha256=sha256_file(resolve_repo_path(args.manifest)),
    )
    counts_by_k = {
        str(k): sum(config["k"] == k for config in manifest["configs"])
        for k in sorted({config["k"] for config in manifest["configs"]})
    }
    report = {
        "schema_version": 2,
        "protocol_amendment_date": protocol["amendment_date"],
        "analysis_provenance": analysis_provenance(args, inventory),
        "data_integrity": {
            "manifest_config_count": len(manifest["configs"]),
            "successful_official_run_count": len(records),
            "counts_by_k": counts_by_k,
            "expected_task_sample_counts": EXPECTED_TASK_SAMPLE_COUNTS,
            **audit_attempts(results_root),
        },
        "primary_permutation": primary_permutation_analysis(records, protocol),
        "paired_document_bootstrap": paired_document_bootstrap(
            records, args.bootstrap_iterations
        ),
        "exploratory_k2": exploratory_k2_analysis(records),
        "dose_response": dose_response_analysis(records, protocol),
        "full_task_aggregate_summary": full_task_aggregate_summary(records),
        "mcnemar_appendix": full_task_mcnemar(records),
        "full_task_bootstrap_cis": full_task_bootstrap_cis(
            records, args.bootstrap_iterations
        ),
        "bi_overlap": protocol["bi_overlap"],
        "canonical_legacy_spearman": {
            model_key: protocol["models"][model_key]["canonical_legacy_spearman"]
            for model_key in MODEL_KEYS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_official_record_export(args.records_output, records, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records_output": str(args.records_output),
                "status": "succeeded",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
