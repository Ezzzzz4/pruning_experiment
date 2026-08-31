import math

import pytest

from experiments.statistics import (
    binary_bootstrap_ci,
    exact_lower_is_better_p,
    holm_adjust,
    mcnemar_exact,
    primary_permutation_analysis,
    weighted_log_likelihood_sum,
    wikitext_word_perplexity,
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
