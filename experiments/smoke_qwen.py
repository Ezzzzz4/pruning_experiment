"""Run the pre-pilot real-model CUDA and cache-equivalence smoke test."""

from __future__ import annotations

import json

import torch

from experiments.benchmark import discover_layer_pool, load_model_and_tokenizer


def main() -> int:
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, provenance = load_model_and_tokenizer("base", "cuda", "float16")
    prefix = tokenizer(
        "Block Influence cache equivalence check.",
        return_tensors="pt",
    ).to("cuda")
    next_token = tokenizer(" Next", return_tensors="pt", add_special_tokens=False)["input_ids"][:, :1].to("cuda")

    with torch.inference_mode():
        unpruned = model(**prefix, use_cache=False).logits

    handler, component, layer_indices = discover_layer_pool(model)
    handler.remove_layers(component, [0], inplace=True)
    with torch.inference_mode():
        full_ids = torch.cat([prefix["input_ids"], next_token], dim=1)
        full_attention = torch.ones_like(full_ids)
        without_cache = model(
            input_ids=full_ids,
            attention_mask=full_attention,
            use_cache=False,
        ).logits[:, -1, :]
        cached_prefix = model(**prefix, use_cache=True)
        if cached_prefix.past_key_values is None:
            raise RuntimeError("use_cache=True did not return past_key_values after pruning.")
        with_cache = model(
            input_ids=next_token,
            past_key_values=cached_prefix.past_key_values,
            use_cache=True,
        ).logits[:, -1, :]

    torch.testing.assert_close(with_cache, without_cache, rtol=1e-3, atol=1e-4)

    result = {
        "model": provenance,
        "component": component,
        "layers_before": len(layer_indices),
        "removed_indices": [0],
        "layers_after": len(handler.get_layers(component)),
        "unpruned_forward_shape": list(unpruned.shape),
        "cache_logits_max_abs_diff": float((with_cache - without_cache).abs().max()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
