"""Generate the frozen conditional-permutation protocol and execution manifest."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from experiments.benchmark import MODEL_REVISIONS, TASKS


AMENDMENT_DATE = "2026-08-18"
LAYER_COUNT = 28
PRIMARY_K = 4
SECONDARY_K = 8
PERMUTATION_SEEDS = tuple(range(3, 23))
FULL_TASK_SEEDS = frozenset(range(3, 8))
EDGE_LAYERS = frozenset({0, 1, 26, 27})
BI_DIR = Path("experiments/bi")
CALIBRATION_PATH = Path("experiments/calibration/wikitext_2_raw_v1_seed1234_n128.jsonl")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bi_bundle(model_key: str) -> dict[str, Any]:
    path = BI_DIR / f"{model_key}.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    expected_revision = MODEL_REVISIONS[model_key]["revision"]
    if bundle.get("model_revision") != expected_revision:
        raise ValueError(
            f"{path} revision {bundle.get('model_revision')!r} does not match {expected_revision!r}."
        )
    expected_indices = {str(idx) for idx in range(LAYER_COUNT)}
    if set(bundle.get("canonical", {})) != expected_indices:
        raise ValueError(f"{path} does not contain exactly {LAYER_COUNT} canonical BI scores.")
    return bundle


def selected_bi_indices(bundle: dict[str, Any], k: int) -> list[int]:
    ordered = sorted(
        range(LAYER_COUNT),
        key=lambda idx: (float(bundle["canonical"][str(idx)]), idx),
    )
    return sorted(ordered[:k])


def shuffled_layer_labels(seed: int) -> list[int]:
    labels = list(range(LAYER_COUNT))
    random.Random(seed).shuffle(labels)
    return labels


def permute_indices(indices: list[int], image_by_source_index: list[int]) -> list[int]:
    return sorted(image_by_source_index[idx] for idx in indices)


def overlap_summary(sets_by_model: dict[str, list[int]]) -> dict[str, Any]:
    models = sorted(sets_by_model)
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            left_set = set(sets_by_model[left])
            right_set = set(sets_by_model[right])
            intersection = sorted(left_set & right_set)
            union = left_set | right_set
            pairwise[f"{left}__{right}"] = {
                "intersection": intersection,
                "intersection_size": len(intersection),
                "jaccard": len(intersection) / len(union),
            }
    common = sorted(set.intersection(*(set(sets_by_model[model]) for model in models)))
    return {"pairwise": pairwise, "three_way_intersection": common}


def protocol_data() -> dict[str, Any]:
    bundles = {model_key: load_bi_bundle(model_key) for model_key in MODEL_REVISIONS}
    bi_indices = {
        model_key: {
            str(k): selected_bi_indices(bundle, k)
            for k in (PRIMARY_K, SECONDARY_K)
        }
        for model_key, bundle in bundles.items()
    }

    permutations = []
    seen_images: set[tuple[int, ...]] = set()
    for seed in PERMUTATION_SEEDS:
        image = shuffled_layer_labels(seed)
        image_key = tuple(image)
        if image_key in seen_images:
            raise ValueError(f"Duplicate layer-label permutation generated for seed {seed}.")
        seen_images.add(image_key)
        by_model: dict[str, Any] = {}
        for model_key in MODEL_REVISIONS:
            by_model[model_key] = {}
            for k in (PRIMARY_K, SECONDARY_K):
                removed = permute_indices(bi_indices[model_key][str(k)], image)
                by_model[model_key][str(k)] = {
                    "removed_indices": removed,
                    "touches_edge": bool(set(removed) & EDGE_LAYERS),
                }
        permutations.append(
            {
                "seed": seed,
                "image_by_source_index": image,
                "full_task_panel": seed in FULL_TASK_SEEDS,
                "by_model": by_model,
            }
        )

    for model_key in MODEL_REVISIONS:
        for k in (PRIMARY_K, SECONDARY_K):
            distinct = {
                tuple(entry["by_model"][model_key][str(k)]["removed_indices"])
                for entry in permutations
            }
            if len(distinct) != len(PERMUTATION_SEEDS):
                raise ValueError(f"Random controls are not distinct for {model_key}, k={k}.")

    return {
        "schema_version": 1,
        "amendment_date": AMENDMENT_DATE,
        "design_status": "frozen_before_first_k4_evaluation",
        "layer_count": LAYER_COUNT,
        "primary": {"k": PRIMARY_K, "metric": "wikitext.word_perplexity"},
        "secondary": {"dose_response_k": SECONDARY_K},
        "metric_directions": {
            "wikitext.word_perplexity": "lower_is_better",
            "arc_challenge.acc_norm": "higher_is_better",
            "piqa.acc_norm": "higher_is_better",
            "winogrande.acc": "higher_is_better",
            "hellaswag.acc_norm": "higher_is_better",
            "lambada_openai.acc": "higher_is_better",
        },
        "analysis": {
            "path": "experiments/statistics.py",
            "bootstrap_seed": 20260818,
            "bootstrap_iterations": 10000,
            "frozen_in_same_commit": True,
        },
        "permutation_seeds": list(PERMUTATION_SEEDS),
        "full_task_seeds": sorted(FULL_TASK_SEEDS),
        "edge_layers": sorted(EDGE_LAYERS),
        "calibration": {
            "path": str(CALIBRATION_PATH),
            "sha256": file_sha256(CALIBRATION_PATH),
        },
        "models": {
            model_key: {
                **MODEL_REVISIONS[model_key],
                "bi_path": str(BI_DIR / f"{model_key}.json"),
                "bi_sha256": file_sha256(BI_DIR / f"{model_key}.json"),
                "canonical_legacy_spearman": bundles[model_key]["rank_correlation_spearman"],
                "bi_indices": bi_indices[model_key],
            }
            for model_key in MODEL_REVISIONS
        },
        "bi_overlap": {
            str(k): overlap_summary(
                {model_key: bi_indices[model_key][str(k)] for model_key in MODEL_REVISIONS}
            )
            for k in (PRIMARY_K, SECONDARY_K)
        },
        "permutations": permutations,
    }


def config_entry(
    model_key: str,
    strategy: str,
    k: int,
    seed: int | None,
    tasks: list[str],
    role: str,
    removed_indices: list[int] | None = None,
    touches_edge: bool | None = None,
    selection_source: str = "conditional_bi_label_permutation",
) -> dict[str, Any]:
    seed_text = "none" if seed is None else str(seed)
    entry: dict[str, Any] = {
        "run_key": f"{model_key}:{strategy}:k{k}:seed{seed_text}",
        "model_key": model_key,
        "model_id": MODEL_REVISIONS[model_key]["model_id"],
        "revision": MODEL_REVISIONS[model_key]["revision"],
        "strategy": strategy,
        "k": k,
        "seed": seed,
        "tasks": tasks,
        "protocol_role": role,
    }
    if removed_indices is not None:
        entry["removed_indices"] = removed_indices
        entry["touches_edge"] = touches_edge
        entry["selection_source"] = selection_source
    return entry


def build_manifest(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = protocol or protocol_data()
    configs: list[dict[str, Any]] = []
    full_tasks = list(TASKS)

    for model_key in MODEL_REVISIONS:
        configs.append(config_entry(model_key, "baseline", 0, None, full_tasks, "baseline"))

    configs.append(config_entry("base", "bi", 2, None, full_tasks, "exploratory_k2"))
    for seed in (0, 1, 2):
        permutation = shuffled_layer_labels(seed)
        removed = sorted(permutation[:2])
        configs.append(
            config_entry(
                "base",
                "random",
                2,
                seed,
                full_tasks,
                "exploratory_k2",
                removed,
                bool(set(removed) & EDGE_LAYERS),
                "legacy_random_permutation_prefix",
            )
        )

    for model_key in MODEL_REVISIONS:
        configs.append(config_entry(model_key, "bi", PRIMARY_K, None, full_tasks, "primary_bi"))
    for permutation in protocol["permutations"]:
        for model_key in MODEL_REVISIONS:
            seed = permutation["seed"]
            selection = permutation["by_model"][model_key][str(PRIMARY_K)]
            tasks = full_tasks if permutation["full_task_panel"] else ["wikitext"]
            configs.append(
                config_entry(
                    model_key,
                    "random",
                    PRIMARY_K,
                    seed,
                    tasks,
                    "primary_random_full" if permutation["full_task_panel"] else "primary_random_wikitext",
                    selection["removed_indices"],
                    selection["touches_edge"],
                )
            )

    for model_key in MODEL_REVISIONS:
        configs.append(
            config_entry(model_key, "bi", SECONDARY_K, None, ["wikitext"], "secondary_k8_bi")
        )
    for permutation in protocol["permutations"]:
        for model_key in MODEL_REVISIONS:
            seed = permutation["seed"]
            selection = permutation["by_model"][model_key][str(SECONDARY_K)]
            configs.append(
                config_entry(
                    model_key,
                    "random",
                    SECONDARY_K,
                    seed,
                    ["wikitext"],
                    "secondary_k8_random",
                    selection["removed_indices"],
                    selection["touches_edge"],
                )
            )

    run_keys = [config["run_key"] for config in configs]
    if len(set(run_keys)) != len(run_keys):
        raise ValueError("Generated manifest contains duplicate run keys.")
    full_count = sum(config["tasks"] == full_tasks for config in configs)
    return {
        "schema_version": 2,
        "amendment_date": AMENDMENT_DATE,
        "protocol_path": "experiments/permutation_protocol.json",
        "harness_revision": "8a07e1110d060de48cfc7a9a7987b7659060b60b",
        "evaluation_seed": 1234,
        "models": MODEL_REVISIONS,
        "tasks": TASKS,
        "grid": {
            "config_count": len(configs),
            "full_task_config_count": full_count,
            "wikitext_only_config_count": len(configs) - full_count,
        },
        "configs": configs,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    protocol = protocol_data()
    write_json(Path("experiments/permutation_protocol.json"), protocol)
    write_json(Path("experiments/experiment_manifest.json"), build_manifest(protocol))
    print(
        json.dumps(
            {
                "protocol": "experiments/permutation_protocol.json",
                "manifest": "experiments/experiment_manifest.json",
                "config_count": build_manifest(protocol)["grid"]["config_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
