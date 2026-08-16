import json
import sys
import types

import pytest

import experiments.benchmark as benchmark


class TinyTokenizer:
    name_or_path = "tiny"
    pad_token = None
    eos_token = "<eos>"


def tiny_model_classes():
    nn = pytest.importorskip("torch.nn")

    class TinyConfig:
        _commit_hash = benchmark.MODEL_REVISIONS["base"]["revision"]
        use_cache = True

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(10)])
            self.config = TinyConfig()

    return TinyModel


def test_build_manifest_has_expected_39_config_grid():
    manifest = benchmark.build_manifest()

    assert manifest["grid"]["config_count"] == 39
    assert len(manifest["configs"]) == 39
    assert {model["revision"] for model in manifest["models"].values()} == {
        "d149729398750b98c0af14eb82c78cfe92750796",
        "a09a35458c702b33eeacc393d103063234e8bc28",
        "ef9926d75ab1d54532f6a30dd5e760355eb9aa4d",
    }
    assert {model["model_id"] for model in manifest["models"].values()} == {
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
    }


def test_random_layer_selection_is_nested_for_same_seed():
    layers = list(range(12))

    removed_2, meta_2 = benchmark.removed_indices_for_config("random", 2, 1, layers)
    removed_4, meta_4 = benchmark.removed_indices_for_config("random", 4, 1, layers)

    assert meta_2["permutation"] == meta_4["permutation"]
    assert set(removed_2).issubset(set(removed_4))


def test_bi_layer_selection_records_full_vector_and_lowest_scores():
    layers = [0, 1, 2, 3]
    scores = {0: 0.4, 1: 0.1, 2: 0.3, 3: 0.2}

    removed, meta = benchmark.removed_indices_for_config("bi", 2, None, layers, scores)

    assert removed == [1, 3]
    assert meta["permutation"] == [1, 3, 2, 0]
    assert meta["bi_scores"] == {"0": 0.4, "1": 0.1, "2": 0.3, "3": 0.2}


def test_hflm_rejects_automatic_batch_probing():
    with pytest.raises(ValueError, match="disabled"):
        benchmark.make_hflm(object(), object(), "auto")


def test_evaluation_context_is_bounded_for_laptop_memory(monkeypatch):
    captured = {}

    class FakeHFLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    hf_module = types.ModuleType("lm_eval.models.huggingface")
    hf_module.HFLM = FakeHFLM
    monkeypatch.setitem(sys.modules, "lm_eval", types.ModuleType("lm_eval"))
    monkeypatch.setitem(sys.modules, "lm_eval.models", types.ModuleType("lm_eval.models"))
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", hf_module)

    benchmark.make_hflm(object(), object(), "4")

    assert captured["batch_size"] == 4
    assert captured["max_length"] == 2048


def test_load_model_verifies_pinned_revision(monkeypatch):
    TinyModel = tiny_model_classes()

    class BadConfig:
        _commit_hash = "wrong"
        use_cache = True

    class BadModel(TinyModel):
        def __init__(self):
            super().__init__()
            self.config = BadConfig()

    transformers = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: BadModel()),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: TinyTokenizer()),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    with pytest.raises(RuntimeError, match="does not match pinned"):
        benchmark.load_model_and_tokenizer("base", "cpu", "float32")


def test_baseline_run_uses_lm_eval_and_writes_outputs(monkeypatch, tmp_path):
    TinyModel = tiny_model_classes()
    eval_calls = []

    monkeypatch.setattr(benchmark, "get_git_dirty", lambda: False)
    monkeypatch.setattr(benchmark, "get_git_sha", lambda: "abc123")
    monkeypatch.setattr(benchmark, "package_version", lambda name: "test-version")
    transformers = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: TinyModel()),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: TinyTokenizer()),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    lm_eval = types.ModuleType("lm_eval")

    def fake_simple_evaluate(**kwargs):
        task = kwargs["tasks"][0]
        eval_calls.append(kwargs)
        return {
            "results": {task: {"acc": 1.0}},
            "samples": {task: [{"doc_id": 0}]},
            "versions": {task: 1},
        }

    lm_eval.simple_evaluate = fake_simple_evaluate
    hf_module = types.ModuleType("lm_eval.models.huggingface")

    class FakeHFLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    hf_module.HFLM = FakeHFLM
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.models", types.ModuleType("lm_eval.models"))
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", hf_module)

    record = benchmark.run_configuration(
        benchmark.RunConfig(
            model_key="base",
            strategy="baseline",
            k=0,
            seed=None,
            output_dir=tmp_path,
            device="cpu",
            dtype="float32",
        ),
        run_id="run-1",
    )

    assert record["status"] == "succeeded"
    assert record["pruning"]["removed_indices"] == []
    assert record["provenance"]["model"]["use_cache"] is False
    assert len(eval_calls) == len(benchmark.TASKS)
    assert all(call["log_samples"] is True for call in eval_calls)

    run_lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(run_lines[0])["run_id"] == "run-1"
    sample_path = tmp_path / "samples" / "base_baseline_k0_seednone__run-1.jsonl"
    sample_lines = sample_path.read_text(encoding="utf-8").splitlines()
    assert len(sample_lines) == len(benchmark.TASKS)
