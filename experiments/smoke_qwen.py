"""Run the pre-pilot real-model CUDA and cache-equivalence smoke test."""

from __future__ import annotations

import json

import torch

from experiments.benchmark import discover_layer_pool, load_model_and_tokenizer


def main() -> int:
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, provenance = load_model_and_tokenizer("base", "cuda", "float16")
    inputs = tokenizer(
        "Block Influence cache equivalence check.",
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        unpruned = model(**inputs, use_cache=False).logits

    handler, component, layer_indices = discover_layer_pool(model)
    handler.remove_layers(component, [2], inplace=True)
    with torch.inference_mode():
        without_cache = model(**inputs, use_cache=False).logits
        with_cache_output = model(**inputs, use_cache=True)
        with_cache = with_cache_output.logits

    torch.testing.assert_close(with_cache, without_cache, rtol=0.0, atol=0.0)
    if with_cache_output.past_key_values is None:
        raise RuntimeError("use_cache=True did not return past_key_values after pruning.")

    result = {
        "model": provenance,
        "component": component,
        "layers_before": len(layer_indices),
        "removed_indices": [2],
        "layers_after": len(handler.get_layers(component)),
        "unpruned_forward_shape": list(unpruned.shape),
        "cache_logits_max_abs_diff": float((with_cache - without_cache).abs().max()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
