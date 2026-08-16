import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.block_influence import compute_block_influence


class AddFirstHidden(nn.Module):
    def __init__(self, *, tuple_output=False):
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, hidden, attention_mask=None):
        delta = torch.tensor([1.0, 0.0], device=hidden.device, dtype=hidden.dtype)
        output = hidden + delta
        if self.tuple_output:
            return output, "cache"
        return output


class FlattenHidden(nn.Module):
    def forward(self, hidden, attention_mask=None):
        return hidden.flatten(start_dim=1)


class TinyDecoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.embedding = nn.Embedding(3, 2)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                    ]
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        return hidden


class FailingDecoder(TinyDecoder):
    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        self.layers[0](hidden, attention_mask=attention_mask)
        raise RuntimeError("forward failed")


def loader(rows, batch_size=8):
    return torch.utils.data.DataLoader(rows, batch_size=batch_size)


def row(input_ids, attention_mask):
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def test_canonical_uses_hand_computed_masked_token_mean():
    model = TinyDecoder([AddFirstHidden()])
    batches = loader([row([0, 1, 2], [1, 1, 0])])

    scores = compute_block_influence(model, model.layers, batches)

    expected = (1.0 - (1.0 / math.sqrt(2.0))) / 2.0
    assert scores == pytest.approx({0: expected})


def test_canonical_padding_invariance():
    compact = TinyDecoder([AddFirstHidden()])
    padded = TinyDecoder([AddFirstHidden()])

    compact_scores = compute_block_influence(
        compact,
        compact.layers,
        loader([row([0, 1], [1, 1])]),
    )
    padded_scores = compute_block_influence(
        padded,
        padded.layers,
        loader([row([0, 1, 2], [1, 1, 0])]),
    )

    assert padded_scores == pytest.approx(compact_scores)


def test_canonical_weights_partial_batches_by_example_not_batch():
    rows = [
        row([0, 0], [1, 1]),
        row([1, 1], [1, 1]),
        row([2, 2], [1, 1]),
    ]
    single_batch = TinyDecoder([AddFirstHidden()])
    split_batches = TinyDecoder([AddFirstHidden()])

    single_scores = compute_block_influence(
        single_batch,
        single_batch.layers,
        loader(rows, batch_size=3),
    )
    split_scores = compute_block_influence(
        split_batches,
        split_batches.layers,
        loader(rows, batch_size=2),
    )

    token0 = 0.0
    token1 = 1.0 - (1.0 / math.sqrt(2.0))
    token2 = 1.0 - (3.0 / math.sqrt(10.0))
    expected = (token0 + token1 + token2) / 3.0
    assert single_scores == pytest.approx({0: expected})
    assert split_scores == pytest.approx({0: expected})


def test_canonical_cosine_computation_uses_fp32(monkeypatch):
    model = TinyDecoder([AddFirstHidden()]).half()
    seen_dtypes = []
    original = F.cosine_similarity

    def spy(x1, x2, *args, **kwargs):
        seen_dtypes.append((x1.dtype, x2.dtype))
        return original(x1, x2, *args, **kwargs)

    monkeypatch.setattr(F, "cosine_similarity", spy)

    compute_block_influence(model, model.layers, loader([row([0, 1], [1, 1])]))

    assert seen_dtypes == [(torch.float32, torch.float32)]


def test_forward_failure_propagates_and_removes_hooks():
    model = FailingDecoder([AddFirstHidden()])

    with pytest.raises(RuntimeError, match="forward failed"):
        compute_block_influence(model, model.layers, loader([row([0, 1], [1, 1])]))

    assert len(model.layers[0]._forward_hooks) == 0


def test_hooks_are_removed_after_success_with_tuple_block_outputs():
    model = TinyDecoder([AddFirstHidden(tuple_output=True)])
    model.train()

    scores = compute_block_influence(model, model.layers, loader([row([0, 1], [1, 1])]))

    assert scores[0] > 0
    assert len(model.layers[0]._forward_hooks) == 0
    assert model.training is True


def test_legacy_fixture_uses_flattened_fp16_cosine_and_ignores_mask():
    model = TinyDecoder([AddFirstHidden()])

    scores = compute_block_influence(
        model,
        model.layers,
        loader([row([0, 1, 2], [1, 1, 0])]),
        mode="legacy",
    )

    assert scores == pytest.approx({0: 0.095703125})


def test_rejects_unsupported_activation_shapes():
    model = TinyDecoder([FlattenHidden()])

    with pytest.raises(ValueError, match=r"shape \[batch, seq, hidden\]"):
        compute_block_influence(model, model.layers, loader([row([0, 1], [1, 1])]))
