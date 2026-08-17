from experiments import prepare_protocol


def test_protocol_preserves_bi_overlap_and_k_nesting():
    protocol = prepare_protocol.protocol_data()

    assert protocol["permutation_seeds"] == list(range(3, 23))
    assert protocol["models"]["base"]["bi_indices"]["4"] == [14, 15, 16, 17]
    assert protocol["bi_overlap"]["4"]["three_way_intersection"] == [14, 15, 16, 17]

    for permutation in protocol["permutations"]:
        k4_sets = []
        k8_sets = []
        for model_key in prepare_protocol.MODEL_REVISIONS:
            k4 = set(permutation["by_model"][model_key]["4"]["removed_indices"])
            k8 = set(permutation["by_model"][model_key]["8"]["removed_indices"])
            assert len(k4) == 4
            assert len(k8) == 8
            assert k4 < k8
            k4_sets.append(k4)
            k8_sets.append(k8)
        assert len(set.intersection(*k4_sets)) == 4
        assert len(set.intersection(*k8_sets)) == 7


def test_manifest_groups_primary_permutations_across_models():
    manifest = prepare_protocol.build_manifest()
    configs = manifest["configs"]

    assert manifest["grid"] == {
        "config_count": 133,
        "full_task_config_count": 25,
        "wikitext_only_config_count": 108,
    }
    seed_three = [config for config in configs if config.get("seed") == 3 and config["k"] == 4]
    assert [config["model_key"] for config in seed_three] == ["base", "instruct", "math"]
    assert all(config["protocol_role"] == "primary_random_full" for config in seed_three)
    assert all(len(config["removed_indices"]) == 4 for config in seed_three)
