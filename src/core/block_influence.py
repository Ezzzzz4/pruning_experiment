"""Block influence scoring for decoder blocks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_block_influence(
    model: nn.Module,
    layers: Iterable[nn.Module],
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    *,
    mode: str = "canonical",
) -> dict[int, float]:
    """Compute block influence scores for decoder blocks.

    Canonical mode computes token-wise cosine distance over the hidden dimension
    in fp32, masks padding tokens, averages within each example, then averages
    examples. Legacy mode preserves the older flattened-example fp16 cosine
    behavior and ignores the attention mask.
    """

    if mode not in {"canonical", "legacy"}:
        raise ValueError("mode must be 'canonical' or 'legacy'")

    layer_list = list(layers)
    if not layer_list:
        raise ValueError("layers must contain at least one decoder block")

    totals = {idx: 0.0 for idx in range(len(layer_list))}
    counts = {idx: 0 for idx in range(len(layer_list))}
    activations: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    model_device = _model_device(model)
    was_training = model.training

    def make_hook(idx: int):
        def hook(
            module: nn.Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            input_hidden = _first_tensor(inputs, "block input")
            output_hidden = _first_tensor(output, "block output")
            _validate_hidden_pair(input_hidden, output_hidden, idx)
            activations[idx] = (input_hidden.detach(), output_hidden.detach())

        return hook

    try:
        model.eval()
        for idx, layer in enumerate(layer_list):
            hooks.append(layer.register_forward_hook(make_hook(idx)))

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = _batch_tensors(batch, model_device)
                activations.clear()

                model(input_ids=input_ids, attention_mask=attention_mask)

                missing = set(totals) - set(activations)
                if missing:
                    missing_text = ", ".join(str(idx) for idx in sorted(missing))
                    raise RuntimeError(f"missing block activations for layers: {missing_text}")

                for idx in range(len(layer_list)):
                    input_hidden, output_hidden = activations.pop(idx)
                    if mode == "canonical":
                        per_example = _canonical_bi(input_hidden, output_hidden, attention_mask)
                    else:
                        per_example = _legacy_bi(input_hidden, output_hidden)

                    totals[idx] += float(per_example.sum().item())
                    counts[idx] += int(per_example.numel())
    finally:
        activations.clear()
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    if any(count == 0 for count in counts.values()):
        raise RuntimeError("no examples were processed")

    return {
        idx: totals[idx] / counts[idx]
        for idx in range(len(layer_list))
        if counts[idx] > 0
    }


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def _batch_tensors(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("dataloader batches must be mappings")
    if "input_ids" not in batch or "attention_mask" not in batch:
        raise KeyError("batch must contain input_ids and attention_mask")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise TypeError("input_ids and attention_mask must be tensors")
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must have shape [batch, seq]")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have matching shapes")

    return input_ids.to(device), attention_mask.to(device)


def _first_tensor(value: object, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"{name} must be a tensor or a tuple whose first item is a tensor")


def _validate_hidden_pair(
    input_hidden: torch.Tensor,
    output_hidden: torch.Tensor,
    layer_idx: int,
) -> None:
    if input_hidden.ndim != 3 or output_hidden.ndim != 3:
        raise ValueError(f"layer {layer_idx} activations must have shape [batch, seq, hidden]")
    if input_hidden.shape != output_hidden.shape:
        raise ValueError(f"layer {layer_idx} input and output shapes must match")


def _canonical_bi(
    input_hidden: torch.Tensor,
    output_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if attention_mask.shape != input_hidden.shape[:2]:
        raise ValueError("attention_mask must match activation batch and sequence dimensions")

    mask = attention_mask.to(device=input_hidden.device).bool()
    token_counts = mask.sum(dim=1)
    if torch.any(token_counts == 0):
        raise ValueError("attention_mask must include at least one token per example")

    cosine = F.cosine_similarity(input_hidden.float(), output_hidden.float(), dim=-1)
    distance = (1.0 - cosine) * mask.float()
    return distance.sum(dim=1) / token_counts.float()


def _legacy_bi(input_hidden: torch.Tensor, output_hidden: torch.Tensor) -> torch.Tensor:
    input_flat = input_hidden.to(torch.float16).reshape(input_hidden.shape[0], -1)
    output_flat = output_hidden.to(torch.float16).reshape(output_hidden.shape[0], -1)
    return 1.0 - F.cosine_similarity(input_flat, output_flat, dim=1)
