from pathlib import Path
import json

import numpy as np
import pytest

from experiments.visualize import edge_touch_by_seed, expected_figure_paths, rank_profile, load_exported_records


def test_expected_figure_paths_are_deterministic():
    output_dir = Path("figures")

    names = [path.name for path in expected_figure_paths(output_dir)]

    assert names == [
        "primary_k4_permutation_ranking.png",
        "k4_wikitext_ppl_by_model.png",
        "k8_dose_response.png",
        "canonical_vs_legacy_bi.png",
        "k4_full_task_secondary_outcomes.png",
    ]


def test_edge_touch_by_seed_collapses_across_models():
    protocol = {
        "permutations": [
            {
                "seed": 3,
                "by_model": {
                    "base": {"4": {"touches_edge": False}},
                    "instruct": {"4": {"touches_edge": True}},
                    "math": {"4": {"touches_edge": False}},
                },
            },
            {
                "seed": 4,
                "by_model": {
                    "base": {"4": {"touches_edge": False}},
                    "instruct": {"4": {"touches_edge": False}},
                    "math": {"4": {"touches_edge": False}},
                },
            },
        ]
    }

    assert edge_touch_by_seed(protocol, 4) == {3: True, 4: False}


def test_rank_profile_ranks_lowest_score_first():
    layers, ranks = rank_profile({"2": 0.3, "0": 0.2, "1": 0.1})

    assert layers == [0, 1, 2]
    np.testing.assert_array_equal(ranks, [2.0, 1.0, 3.0])


def test_rank_profile_averages_ties():
    _, ranks = rank_profile({"0": 0.1, "1": 0.1, "2": 0.2})

    np.testing.assert_array_equal(ranks, [1.5, 1.5, 3.0])


def test_figures_can_load_tracked_export_without_raw_logs(tmp_path):
    path = tmp_path / "runs.jsonl"
    record = {"provenance": {"run_key": "base:bi:k4:seednone"},
              "status": "succeeded", "config": {"official_run": True}}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert load_exported_records(path) == {"base:bi:k4:seednone": record}

    path.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate run key"):
        load_exported_records(path)
