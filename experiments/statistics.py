"""Frozen statistical analysis for the 2026-08-18 permutation protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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

    edge_free_seeds = [
        seed
        for seed in PERMUTATION_SEEDS
        if not any(
            protocol_selection(protocol, seed, model_key, PRIMARY_K)["touches_edge"]
            for model_key in MODEL_KEYS
        )
    ]
    edge_free_p = exact_lower_is_better_p(
        bi_statistic, (random_statistics[seed] for seed in edge_free_seeds)
    )

    model_specific = {}
    for model_key in MODEL_KEYS:
        random_values = [random_ppl[seed][model_key] for seed in PERMUTATION_SEEDS]
        model_specific[model_key] = {
            "baseline_ppl": baselines[model_key],
            "bi_ppl": bi_ppl[model_key],
            "bi_log_degradation": bi_log_degradation[model_key],
            "bi_rank_lower_is_better": rank_by_model[model_key]["bi"],
            "descriptive_exact_p": exact_lower_is_better_p(bi_ppl[model_key], random_values),
            "random_ppl": {str(seed): random_ppl[seed][model_key] for seed in PERMUTATION_SEEDS},
        }

    return {
        "primary": {
            "statistic": "mean_model_log_ppl_ratio_to_baseline",
            "bi": bi_statistic,
            "random": {str(seed): random_statistics[seed] for seed in PERMUTATION_SEEDS},
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
            "edge_free_seeds": edge_free_seeds,
            "edge_free_count": len(edge_free_seeds),
            "bi_rank_p_among_edge_free": edge_free_p,
        },
        "model_specific_descriptive": model_specific,
    }


def load_task_samples(record: dict[str, Any], task: str) -> dict[str, dict[str, Any]]:
    path = Path(record["sample_log_path"])
    samples: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["task"] != task:
                continue
            sample = row["sample"]
            key = str(sample.get("doc_hash") or sample["doc_id"])
            if key in samples:
                raise ValueError(f"Duplicate {task} sample key {key} in {path}.")
            samples[key] = sample
    if not samples:
        raise ValueError(f"No {task} samples found in {path}.")
    return samples


def wikitext_document_arrays(record: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    samples = load_task_samples(record, "wikitext")
    keys = sorted(samples)
    log_likelihood = np.array(
        [
            float(decode_non_finite_float(samples[key]["word_perplexity"][0]))
            for key in keys
        ]
    )
    words = np.array([float(samples[key]["word_perplexity"][1]) for key in keys])
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
    p_value = 1.0 if discordant == 0 else binomtest(
        bi_only, discordant, p=0.5, alternative="greater"
    ).pvalue
    return {
        "bi_only_correct": bi_only,
        "random_only_correct": random_only,
        "discordant": discordant,
        "one_sided_p": float(p_value),
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
            for seed in FULL_TASK_SEEDS:
                random_record = require_records(
                    records, [f"{model_key}:random:k{PRIMARY_K}:seed{seed}"]
                )[0]
                comparison = mcnemar_exact(
                    bi_samples, load_task_samples(random_record, task), metric
                )
                raw[str(seed)] = comparison
                p_values[str(seed)] = comparison["one_sided_p"]
            adjusted = holm_adjust(p_values)
            for seed in FULL_TASK_SEEDS:
                raw[str(seed)]["holm_adjusted_p"] = adjusted[str(seed)]
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen pruning analysis.")
    parser.add_argument("--results-root", type=Path, default=Path("results/lm_eval"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/permutation_protocol.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/lm_eval/analysis/summary.json")
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = read_json(args.protocol)
    records = load_successful_official_records(args.results_root)
    report = {
        "schema_version": 1,
        "protocol_amendment_date": protocol["amendment_date"],
        "primary_permutation": primary_permutation_analysis(records, protocol),
        "paired_document_bootstrap": paired_document_bootstrap(
            records, args.bootstrap_iterations
        ),
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
    print(json.dumps({"output": str(args.output), "status": "succeeded"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
