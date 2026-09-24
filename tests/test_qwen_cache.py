import pytest
import torch
import torch.nn as nn

from src.handlers.universal_handler import UniversalHandler


transformers = pytest.importorskip("transformers")
Qwen2Config = transformers.Qwen2Config
Qwen2ForCausalLM = transformers.Qwen2ForCausalLM


def tiny_qwen(num_layers=4):
    torch.manual_seed(1234)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    return model.eval()


def cached_continuation_logits(model, prefix, next_token):
    with torch.no_grad():
        cached_prefix = model(prefix, use_cache=True)
        cached_next = model(
            next_token,
            past_key_values=cached_prefix.past_key_values,
            use_cache=True,
        ).logits[:, -1, :]
        full_next = model(
            torch.cat([prefix, next_token], dim=1),
            use_cache=False,
        ).logits[:, -1, :]
    return cached_next, full_next


def manual_pruned_copy(model, indices):
    import copy

    pruned = copy.deepcopy(model)
    kept = [
        layer for idx, layer in enumerate(pruned.model.layers)
        if idx not in set(indices)
    ]
    pruned.model.layers = nn.ModuleList(kept)
    return pruned.eval()


@pytest.mark.parametrize("indices", [[0], [1], [3], [0, 2]])
def test_qwen_pruned_cached_continuation_matches_full_forward(indices):
    model = tiny_qwen()
    handler = UniversalHandler(model, verbose=False)
    handler.remove_layers("main", indices, inplace=True)

    assert model.config.num_hidden_layers == len(model.model.layers)
    assert [layer.self_attn.layer_idx for layer in model.model.layers] == list(
        range(len(model.model.layers))
    )

    prefix = torch.tensor([[1, 7, 11, 13]])
    next_token = torch.tensor([[17]])
    cached_next, full_next = cached_continuation_logits(model, prefix, next_token)

    torch.testing.assert_close(cached_next, full_next, rtol=1e-4, atol=1e-5)


def test_qwen_copy_prune_reindexes_cache_without_mutating_original():
    model = tiny_qwen()
    original_indices = [layer.self_attn.layer_idx for layer in model.model.layers]
    handler = UniversalHandler(model, verbose=False)

    pruned = handler.remove_layer("main", 0, inplace=False)

    assert model.config.num_hidden_layers == 4
    assert [layer.self_attn.layer_idx for layer in model.model.layers] == original_indices
    assert pruned.config.num_hidden_layers == 3
    assert [layer.self_attn.layer_idx for layer in pruned.model.layers] == [0, 1, 2]

    prefix = torch.tensor([[2, 3, 5, 7]])
    next_token = torch.tensor([[11]])
    cached_next, full_next = cached_continuation_logits(pruned, prefix, next_token)

    torch.testing.assert_close(cached_next, full_next, rtol=1e-4, atol=1e-5)


def test_qwen_no_cache_logits_match_manual_layer_removal():
    model = tiny_qwen()
    manual = manual_pruned_copy(model, [0, 2])
    handler = UniversalHandler(model, verbose=False)
    handler.remove_layers("main", [0, 2], inplace=True)

    input_ids = torch.tensor([[1, 7, 11, 13, 17]])
    with torch.no_grad():
        handled_logits = model(input_ids, use_cache=False).logits
        manual_logits = manual(input_ids, use_cache=False).logits

    torch.testing.assert_close(handled_logits, manual_logits, rtol=0.0, atol=0.0)
