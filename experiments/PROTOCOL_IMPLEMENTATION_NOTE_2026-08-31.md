# Protocol implementation note — 2026-08-31

The frozen layer selections, tasks, metrics, execution order, and statistical comparisons remain unchanged.

`instruct:random:k4:seed13` first failed with a CUDA illegal-memory-access error. Two exact retries reached WikiText evaluation but produced non-finite floating-point values while removing the same frozen layers `[2, 6, 11, 22]`. Strict JSON serialization rejected those values, so neither retry could become a successful evidence record.

The runner now preserves non-finite values as explicit strict-JSON sentinel objects:

```json
{"__non_finite_float__": "positive_infinity"}
{"__non_finite_float__": "negative_infinity"}
{"__non_finite_float__": "nan"}
```

This is a representation change, not a metric substitution. Positive-infinite WikiText perplexity ranks as the worst possible outcome. Negative-infinite document log likelihood produces positive-infinite degradation. NaN and sign-inconsistent infinities remain invalid.

The paired document bootstrap avoids the undefined product `0 * -inf`: a resample receives negative-infinite total log likelihood only when it includes a catastrophic document. All attempts remain in the append-only artifacts.

The Windows environment also pins `pandas==2.2.3` and `pyarrow==21.0.0`. Windows Application Control blocked the newer transitive binary wheels; these versions load under the active policy. Model revisions, the harness revision, CUDA/PyTorch, and evaluation code are otherwise unchanged.
