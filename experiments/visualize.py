"""Generate figures for the frozen lm-eval pruning experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from experiments.statistics import (
    FULL_TASK_SEEDS,
    MC_PRIMARY_METRICS,
    MODEL_KEYS,
    PERMUTATION_SEEDS,
    PRIMARY_K,
    load_successful_official_records,
    read_json,
    require_records,
    resolve_repo_path,
    wikitext_word_perplexity,
)


MODEL_LABELS = {
    "base": "Qwen2.5-7B",
    "instruct": "Qwen2.5-7B-Instruct",
    "math": "Qwen2.5-Math-7B-Instruct",
}
FIGURE_NAMES = (
    "primary_k4_permutation_ranking.png",
    "k4_wikitext_ppl_by_model.png",
    "k8_dose_response.png",
    "canonical_vs_legacy_bi.png",
    "k4_full_task_secondary_outcomes.png",
)
TASK_LABELS = {
    "arc_challenge": "ARC-C",
    "piqa": "PIQA",
    "winogrande": "Winogrande",
    "hellaswag": "HellaSwag",
    "lambada_openai": "Lambada",
}


def set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def expected_figure_paths(output_dir: Path) -> list[Path]:
    return [output_dir / name for name in FIGURE_NAMES]


def edge_touch_by_seed(protocol: dict[str, Any], k: int) -> dict[int, bool]:
    output: dict[int, bool] = {}
    for entry in protocol["permutations"]:
        seed = int(entry["seed"])
        output[seed] = any(
            entry["by_model"][model_key][str(k)]["touches_edge"]
            for model_key in MODEL_KEYS
        )
    return output


def plot_primary_permutation(summary: dict[str, Any], protocol: dict[str, Any], output: Path) -> None:
    primary = summary["primary_permutation"]["primary"]
    edge_touch = edge_touch_by_seed(protocol, PRIMARY_K)
    candidates = [("BI", float(primary["bi"]), False)]
    candidates.extend(
        (f"seed {seed}", float(primary["random"][str(seed)]), edge_touch[seed])
        for seed in PERMUTATION_SEEDS
    )
    candidates.sort(key=lambda item: item[1])

    labels = [item[0] for item in candidates]
    values = [item[1] for item in candidates]
    colors = [
        "#1f77b4" if label == "BI" else "#d95f02" if touches_edge else "#4daf4a"
        for label, _, touches_edge in candidates
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("mean log(PPL pruned / baseline)")
    ax.set_title("Primary k=4 WikiText permutation ranking")
    ax.text(
        0.99,
        0.96,
        f"one-sided exact p = {primary['one_sided_exact_p']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="BI"),
            plt.Rectangle((0, 0), 1, 1, color="#4daf4a", label="random, edge-free"),
            plt.Rectangle((0, 0), 1, 1, color="#d95f02", label="random, touches edge"),
        ],
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_model_k4_ppl(summary: dict[str, Any], output: Path) -> None:
    model_specific = summary["primary_permutation"]["model_specific_descriptive"]
    fig, axes = plt.subplots(1, len(MODEL_KEYS), figsize=(13, 4), sharey=False)

    for ax, model_key in zip(axes, MODEL_KEYS, strict=True):
        data = model_specific[model_key]
        seeds = np.array(PERMUTATION_SEEDS)
        random_values = np.array(
            [float(data["random_ppl"][str(seed)]) for seed in PERMUTATION_SEEDS]
        )
        ax.scatter(seeds, random_values, color="#808080", s=24)
        ax.axhline(float(data["baseline_ppl"]), color="#333333", linestyle="--", label="baseline")
        ax.axhline(float(data["bi_ppl"]), color="#1f77b4", linewidth=2, label="BI")
        ax.set_yscale("log")
        ax.set_title(MODEL_LABELS[model_key])
        ax.set_xlabel("frozen permutation seed")
        ax.set_xticks([3, 6, 9, 12, 15, 18, 21])
        ax.set_ylabel("WikiText word perplexity")
        ax.legend(frameon=True)

    fig.suptitle("k=4 WikiText PPL by model")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def log_ratio(record: dict[str, Any], baseline_ppl: float) -> float:
    value = wikitext_word_perplexity(record)
    if math.isinf(value):
        return math.inf
    return math.log(value / baseline_ppl)


def plot_k8_dose_response(records: dict[str, dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, len(MODEL_KEYS), figsize=(13, 4), sharey=False)

    for ax, model_key in zip(axes, MODEL_KEYS, strict=True):
        baseline = wikitext_word_perplexity(
            require_records(records, [f"{model_key}:baseline:k0:seednone"])[0]
        )
        bi_k4 = log_ratio(
            require_records(records, [f"{model_key}:bi:k4:seednone"])[0], baseline
        )
        bi_k8 = log_ratio(
            require_records(records, [f"{model_key}:bi:k8:seednone"])[0], baseline
        )
        random_k4 = [
            log_ratio(
                require_records(records, [f"{model_key}:random:k4:seed{seed}"])[0],
                baseline,
            )
            for seed in PERMUTATION_SEEDS
        ]
        random_k8 = [
            log_ratio(
                require_records(records, [f"{model_key}:random:k8:seed{seed}"])[0],
                baseline,
            )
            for seed in PERMUTATION_SEEDS
        ]

        ax.boxplot([random_k4, random_k8], tick_labels=["random k=4", "random k=8"])
        ax.scatter([1, 2], [bi_k4, bi_k8], color="#1f77b4", s=48, zorder=3, label="BI")
        ax.set_title(MODEL_LABELS[model_key])
        ax.set_ylabel("log(PPL pruned / baseline)")
        ax.legend(frameon=True)

    fig.suptitle("Dose response on WikiText")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def rank_profile(scores: dict[str, float]) -> tuple[list[int], np.ndarray]:
    layers = sorted(int(layer) for layer in scores)
    values = np.array([float(scores[str(layer)]) for layer in layers])
    return layers, rankdata(values, method="average")


def plot_bi_legacy_profiles(summary: dict[str, Any], protocol: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(len(MODEL_KEYS), 1, figsize=(10, 8), sharex=True, sharey=True)

    for ax, model_key in zip(axes, MODEL_KEYS, strict=True):
        bi_path = resolve_repo_path(protocol["models"][model_key]["bi_path"])
        bundle = read_json(bi_path)
        canonical_layers, canonical_ranks = rank_profile(bundle["canonical"])
        legacy_layers, legacy_ranks = rank_profile(bundle["legacy"])
        if canonical_layers != legacy_layers:
            raise ValueError(f"Canonical and legacy layers differ in {bi_path}.")
        rho = summary["canonical_legacy_spearman"][model_key]
        ax.plot(canonical_layers, canonical_ranks, marker="o", markersize=3, label="canonical")
        ax.plot(legacy_layers, legacy_ranks, marker="s", markersize=3, label="legacy")
        ax.set_title(f"{MODEL_LABELS[model_key]} (Spearman rho = {rho:.3f})")
        ax.set_ylabel("rank, lower = pruned earlier")
        ax.invert_yaxis()
        ax.legend(frameon=True, loc="lower right")

    axes[-1].set_xlabel("layer index")
    fig.suptitle("Canonical vs legacy BI layer rankings")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_full_task_secondary(summary: dict[str, Any], output: Path) -> None:
    cis = summary["full_task_bootstrap_cis"]
    tasks = list(MC_PRIMARY_METRICS)
    x = np.arange(len(tasks))
    fig, axes = plt.subplots(len(MODEL_KEYS), 1, figsize=(11, 8), sharex=True, sharey=True)

    for ax, model_key in zip(axes, MODEL_KEYS, strict=True):
        baseline_key = f"{model_key}:baseline:k0:seednone"
        bi_key = f"{model_key}:bi:k{PRIMARY_K}:seednone"
        baseline = {
            task: float(cis[model_key][baseline_key][task]["estimate"])
            for task in tasks
        }
        bi_delta = np.array(
            [(float(cis[model_key][bi_key][task]["estimate"]) - baseline[task]) * 100 for task in tasks]
        )

        offsets = np.linspace(-0.18, 0.18, len(FULL_TASK_SEEDS))
        for index, (seed, offset) in enumerate(zip(FULL_TASK_SEEDS, offsets, strict=True)):
            random_key = f"{model_key}:random:k{PRIMARY_K}:seed{seed}"
            random_delta = np.array(
                [
                    (float(cis[model_key][random_key][task]["estimate"]) - baseline[task]) * 100
                    for task in tasks
                ]
            )
            ax.scatter(
                x + offset,
                random_delta,
                color="#9e9e9e",
                s=22,
                alpha=0.8,
                label="random controls" if index == 0 else None,
            )

        ax.scatter(x, bi_delta, color="#1f77b4", s=46, zorder=3, label="BI")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(MODEL_LABELS[model_key])
        ax.set_ylabel("accuracy change vs baseline (pp)")
        ax.legend(frameon=True, loc="lower left")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([TASK_LABELS[task] for task in tasks], rotation=20, ha="right")
    fig.suptitle("Secondary k=4 full-task outcomes")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate confirmatory experiment figures.")
    parser.add_argument("--summary", type=Path, default=Path("results/confirmatory/summary.json"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/permutation_protocol.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results/lm_eval"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/confirmatory/figures"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_style()
    summary = read_json(args.summary)
    protocol = read_json(args.protocol)
    records = load_successful_official_records(args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_primary_permutation(
        summary,
        protocol,
        args.output_dir / "primary_k4_permutation_ranking.png",
    )
    plot_model_k4_ppl(summary, args.output_dir / "k4_wikitext_ppl_by_model.png")
    plot_k8_dose_response(records, args.output_dir / "k8_dose_response.png")
    plot_bi_legacy_profiles(summary, protocol, args.output_dir / "canonical_vs_legacy_bi.png")
    plot_full_task_secondary(
        summary,
        args.output_dir / "k4_full_task_secondary_outcomes.png",
    )

    print(
        json.dumps(
            {
                "figures": [str(path) for path in expected_figure_paths(args.output_dir)],
                "output_dir": str(args.output_dir),
                "status": "succeeded",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
