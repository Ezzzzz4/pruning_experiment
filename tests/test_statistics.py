import json
import math
from decimal import Decimal

import pytest

from experiments.statistics import (
    binary_bootstrap_ci,
    exact_lower_is_better_p,
    holm_adjust,
    load_task_samples,
    mcnemar_exact,
    primary_permutation_analysis,
    validate_official_records,
    weighted_log_likelihood_sum,
    wikitext_word_perplexity,
    write_official_record_export,
)


def ppl_record(run_key, value):
    return {
        "provenance": {"run_key": run_key},
        "results": {"results": {"wikitext": {"word_perplexity,none": value}}},
    }


def test_exact_p_has_predeclared_minimum_with_twenty_controls():
    assert exact_lower_is_better_p(0.0, range(1, 21)) == pytest.approx(1 / 21)
    assert exact_lower_is_better_p(2.0, range(1, 21)) == pytest.approx(3 / 21)


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})

    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_mcnemar_exact_counts_paired_disagreements():
    bi = {
        "a": {"acc": 1.0},
        "b": {"acc": 1.0},
        "c": {"acc": 0.0},
        "d": {"acc": 1.0},
    }
    random_samples = {
        "a": {"acc": 0.0},
        "b": {"acc": 0.0},
        "c": {"acc": 1.0},
        "d": {"acc": 1.0},
    }

    result = mcnemar_exact(bi, random_samples, "acc")

    assert result["bi_only_correct"] == 2
    assert result["random_only_correct"] == 1
    assert result["discordant"] == 3
    assert math.isclose(result["one_sided_p"], 0.5)


def test_mcnemar_rejects_non_finite_or_non_binary_correctness():
    bi = {"a": {"acc": {"__non_finite_float__": "nan"}}}
    random_samples = {"a": {"acc": 0.0}}

    with pytest.raises(ValueError, match="Invalid binary metric"):
        mcnemar_exact(bi, random_samples, "acc")

    with pytest.raises(ValueError, match="Invalid binary metric"):
        mcnemar_exact({"a": {"acc": 0.5}}, random_samples, "acc")


def test_mcnemar_preserves_extreme_probability_as_decimal():
    bi = {str(index): {"acc": 1} for index in range(1_100)}
    random_samples = {str(index): {"acc": 0} for index in range(1_100)}

    result = mcnemar_exact(bi, random_samples, "acc")

    assert result["one_sided_p"] == 0.0
    assert Decimal(result["one_sided_p_decimal"]) > 0


def test_load_task_samples_keys_by_doc_id_and_doc_hash(tmp_path):
    sample_log = tmp_path / "samples.jsonl"
    sample_log.write_text(
        "\n".join(
            [
                '{"task": "lambada_openai", "sample": {"doc_id": 0, "doc_hash": "same", "acc": 1}}',
                '{"task": "lambada_openai", "sample": {"doc_id": 1, "doc_hash": "same", "acc": 0}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    record = {"sample_log_path": str(sample_log)}

    samples = load_task_samples(record, "lambada_openai")

    assert sorted(samples) == ['[0,"same"]', '[1,"same"]']


def test_load_task_samples_requires_document_id_and_content_hash(tmp_path):
    sample_log = tmp_path / "samples.jsonl"
    sample_log.write_text(
        json.dumps(
            {
                "task": "piqa",
                "sample": {"doc_id": 3, "doc_hash": None, "acc_norm": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing stable piqa sample identity"):
        load_task_samples({"sample_log_path": str(sample_log)}, "piqa")


def test_mcnemar_rejects_same_document_id_with_changed_content(tmp_path):
    first_log = tmp_path / "first.jsonl"
    second_log = tmp_path / "second.jsonl"
    first_log.write_text(
        '{"task": "piqa", "sample": {"doc_id": 3, "doc_hash": "original", "acc_norm": 1}}\n',
        encoding="utf-8",
    )
    second_log.write_text(
        '{"task": "piqa", "sample": {"doc_id": 3, "doc_hash": "changed", "acc_norm": 0}}\n',
        encoding="utf-8",
    )

    first = load_task_samples({"sample_log_path": str(first_log)}, "piqa")
    second = load_task_samples({"sample_log_path": str(second_log)}, "piqa")

    with pytest.raises(ValueError, match="sample keys do not align"):
        mcnemar_exact(first, second, "acc_norm")


def test_official_record_export_uses_manifest_order(tmp_path):
    path = tmp_path / "official_runs.jsonl"
    records = {
        "second": {"run_id": "2", "value": 2},
        "first": {"run_id": "1", "value": 1},
    }
    manifest = {"configs": [{"run_key": "first"}, {"run_key": "second"}]}

    write_official_record_export(path, records, manifest)

    exported = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["run_id"] for record in exported] == ["1", "2"]


def test_official_record_validation_checks_manifest_and_sample_counts(tmp_path):
    sample_log = tmp_path / "samples.jsonl"
    run_id = "run-1"
    sample_log.write_text(
        "".join(
            json.dumps(
                {
                    "run_id": run_id,
                    "task": "wikitext",
                    "sample": {
                        "doc_id": index,
                        "doc_hash": f"hash-{index}",
                        "word_perplexity": [-1.0, 1.0],
                    },
                }
            )
            + "\n"
            for index in range(62)
        ),
        encoding="utf-8",
    )
    run_key = "base:random:k4:seed3"
    config = {
        "run_key": run_key,
        "model_key": "base",
        "model_id": "model",
        "revision": "revision",
        "strategy": "random",
        "k": 4,
        "seed": 3,
        "removed_indices": [1, 2, 3, 4],
        "selection_source": "frozen",
        "tasks": ["wikitext"],
    }
    record = {
        "run_id": run_id,
        "status": "succeeded",
        "sample_log_path": str(sample_log),
        "config": {
            "model_key": "base",
            "strategy": "random",
            "k": 4,
            "seed": 3,
            "official_run": True,
        },
        "provenance": {
            "run_key": run_key,
            "tasks": ["wikitext"],
            "limit": None,
            "batch_size": "4",
            "harness_expected_sha": "harness",
            "harness_installed_sha": "harness",
            "code_sha": "code",
            "device": "cuda",
            "worktree_dirty": False,
            "seeds": {"evaluation": 1234, "strategy": 3},
            "model": {
                "model_id": "model",
                "expected_revision": "revision",
                "loaded_revision": "revision",
                "parameter_dtype": "torch.float16",
            },
        },
        "pruning": {
            "removed_indices": [1, 2, 3, 4],
            "selection_source": "frozen",
        },
        "results": {
            "results": {"wikitext": {"word_perplexity,none": math.e}}
        },
    }
    manifest = {
        "evaluation_seed": 1234,
        "harness_revision": "harness",
        "configs": [config],
    }

    inventory = validate_official_records({run_key: record}, manifest)

    assert inventory[run_key]["sample_counts"] == {"wikitext": 62}
    record["pruning"]["removed_indices"] = [1, 2, 3, 5]
    with pytest.raises(ValueError, match="Removed-index mismatch"):
        validate_official_records({run_key: record}, manifest)

    record["pruning"]["removed_indices"] = [1, 2, 3, 4]
    record["results"]["results"]["wikitext"]["word_perplexity,none"] = 3.0
    with pytest.raises(ValueError, match="aggregate/sample mismatch"):
        validate_official_records({run_key: record}, manifest)


def test_binary_bootstrap_ci_is_reproducible():
    import numpy as np

    first = binary_bootstrap_ci([1, 1, 0, 0], np.random.default_rng(7), 1_000)
    second = binary_bootstrap_ci([1, 1, 0, 0], np.random.default_rng(7), 1_000)

    assert first == second
    assert first["estimate"] == 0.5
    assert first["sample_count"] == 4


def test_primary_analysis_ranks_bi_against_twenty_global_permutations():
    records = {}
    for model_offset, model_key in enumerate(("base", "instruct", "math")):
        baseline_key = f"{model_key}:baseline:k0:seednone"
        bi_key = f"{model_key}:bi:k4:seednone"
        records[baseline_key] = ppl_record(baseline_key, 10 + model_offset)
        records[bi_key] = ppl_record(bi_key, 11 + model_offset)
        for seed in range(3, 23):
            run_key = f"{model_key}:random:k4:seed{seed}"
            records[run_key] = ppl_record(run_key, 12 + model_offset + (seed - 3) / 10)
    protocol = {
        "edge_layers": [0, 1, 26, 27],
        "models": {
            model_key: {"bi_indices": {"4": [10, 11, 12, 13]}}
            for model_key in ("base", "instruct", "math")
        },
        "permutations": [
            {
                "seed": seed,
                "by_model": {
                    model_key: {"4": {"touches_edge": False}}
                    for model_key in ("base", "instruct", "math")
                },
            }
            for seed in range(3, 23)
        ],
    }
    catastrophic_key = "instruct:random:k4:seed3"
    records[catastrophic_key] = ppl_record(
        catastrophic_key,
        {"__non_finite_float__": "positive_infinity"},
    )

    result = primary_permutation_analysis(records, protocol)

    assert result["primary"]["one_sided_exact_p"] == pytest.approx(1 / 21)
    assert result["primary"]["random"]["3"] == math.inf
    assert result["model_specific_descriptive"]["instruct"]["random_ppl"]["3"] == math.inf
    assert result["robustness"]["one_sided_exact_p"] == pytest.approx(1 / 21)
    assert result["edge_diagnostic"]["edge_free_count"] == 20


def test_wikitext_perplexity_decodes_positive_infinity_as_worst_value():
    record = ppl_record(
        "base:random:k4:seed13",
        {"__non_finite_float__": "positive_infinity"},
    )

    assert wikitext_word_perplexity(record) == math.inf


def test_weighted_log_likelihood_sum_handles_unselected_negative_infinity():
    import numpy as np

    counts = np.array([[1, 0], [0, 1], [1, 1]])
    values = np.array([-2.0, -math.inf])

    totals = weighted_log_likelihood_sum(counts, values)

    assert totals[0] == -2.0
    assert totals[1] == -math.inf
    assert totals[2] == -math.inf
