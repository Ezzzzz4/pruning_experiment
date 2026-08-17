# Protocol amendment — 2026-08-18

Status: frozen before the first `k=4` evaluation.

This amendment replaces the original `random×3` confirmatory grid. It was triggered by the large spread observed between the first two random `k=2` controls for Qwen2.5-7B. Those observations are exploratory, not confirmatory. They remain part of the evidence and will be reported regardless of later results.

Seeds 0–2 are excluded from the amended primary analysis. The original random permutations are nested across `k`, so observing `k=2` revealed partial information about their `k=4` subsets.

## Fixed inputs

- Models and immutable revisions are recorded in `experiments/permutation_protocol.json`.
- All three models have 28 transformer blocks.
- Canonical and legacy BI vectors are stored under `experiments/bi/`.
- BI uses the same 128-document WikiText calibration sample for every model.
- The calibration path and SHA-256 are recorded in the protocol JSON.
- Canonical BI determines layer removal. Legacy BI is a sensitivity analysis only.

Canonical BI-selected layers:

| Model | k=4 | k=8 |
|---|---|---|
| Qwen2.5-7B | 14, 15, 16, 17 | 12, 13, 14, 15, 16, 17, 18, 25 |
| Qwen2.5-7B-Instruct | 14, 15, 16, 17 | 12, 13, 14, 15, 16, 17, 18, 20 |
| Qwen2.5-Math-7B-Instruct | 14, 15, 16, 17 | 11, 12, 13, 14, 15, 16, 17, 18 |

## Conditional random controls

Twenty new global layer-label permutations use seeds 3–22. A permutation is a bijection of layer labels 0–27. For permutation `π`, model `m`, and pruning level `k`, the random control is:

`R[j,m,k] = π[j](BI[m,k])`.

The same permutation is applied to every model and to both `k=4` and `k=8`. This guarantees:

- every model-specific control is marginally uniform over all `k`-subsets;
- the observed pairwise and three-way overlap structure of the BI selections is preserved exactly;
- `k=4` remains nested inside `k=8`;
- BI is the identity-permutation member of the same conditional permutation family.

The test is conditional on the observed overlap structure of the three BI selections. It asks whether the selected layer positions are unusually good within this conditional family; it does not estimate an unconditional population of possible fine-tuning recipes.

The exact 20 permutations and every removed-index list are frozen in `experiments/permutation_protocol.json`. Seeds alone are not the evidence source.

## Evaluation grid

- Baseline: all six tasks for all three models.
- Exploratory `k=2`: the already completed base-model BI run and random seeds 0–2, all six tasks.
- Primary `k=4`:
  - BI on all six tasks for every model;
  - permutations 3–7 on all six tasks for every model;
  - permutations 8–22 on WikiText only for every model.
- Secondary `k=8` dose response:
  - BI and all permutations 3–22 on WikiText only for every model.

The frozen manifest contains 133 records: 25 full-task records and 108 WikiText-only records. Most records are not full harness runs.

## Primary estimand and test

The primary metric is WikiText word perplexity at `k=4`; lower is better.

For full-task secondary outcomes, higher is better for ARC-Challenge `acc_norm`, PIQA `acc_norm`, Winogrande `acc`, HellaSwag `acc_norm`, and Lambada `acc`.

For candidate selection object `s`:

`T(s) = mean_m(log(PPL[s,m] / PPL[baseline,m]))`.

Lower `T` is better. The one-sided exact Monte Carlo p-value is:

`p = (1 + count(T(random_j) <= T(BI))) / 21`.

Random outcomes tied with BI count as “not worse” and enter the numerator. The smallest attainable p-value is `1/21 ≈ 0.047619`.

The predeclared robustness statistic replaces each model’s log-ratio values by within-model ranks and averages the three ranks. Its p-value uses the same conservative tie rule.

Model-specific BI percentiles are secondary and descriptive. With one checkpoint per training regime and one deterministic BI selection per checkpoint, the experiment does not support a causal significance test for dependence on fine-tuning. Differences across checkpoints will be reported as descriptive effect sizes and percentiles.

## Edge-layer diagnostic

Edge layers are fixed as `{0, 1, 26, 27}` before primary evaluation.

For every random control, the protocol records whether any selected set in its three-model object touches an edge layer. The report will always show:

- performance distributions split by edge contact;
- the number of edge-free controls;
- BI’s rank among edge-free controls, if BI itself is edge-free.

The edge-free rank is a predeclared secondary diagnostic with its actual denominator. It does not replace the primary test.

## Sampling uncertainty and task-level analyses

- The primary permutation test treats the layer-selection object as the exchangeable unit.
- WikiText document uncertainty is evaluated separately by jointly resampling the same 62 documents across BI, baseline, and all random controls. Pairing is preserved.
- Bootstrap confidence intervals use a fixed analysis seed and are sensitivity estimates, not replacements for the permutation test.
- Multiple-choice McNemar comparisons are appendix analyses for BI versus each of the five full-task random controls. Holm correction is applied within each model/task family.
- Bootstrap intervals and McNemar tests cannot substitute for variation across layer selections.

## Failures, retries, and incomplete execution

- A technical failure means an exception, OOM, non-finite output, corrupt/missing sample log, or sample-count mismatch.
- Technical failures are retried with exactly the same model, permutation, removed indices, tasks, and software revision.
- A result is never retried because its metric appears surprising or unfavorable.
- Every attempt remains represented by its append-only artifacts; interrupted attempts without a terminal JSONL record are retained outside the evidence index.
- Execution order is fixed by the manifest and grouped by permutation across the three models.
- Confirmatory inference requires all 20 primary permutation objects. If fewer complete by the cutoff, available data are reported as incomplete/exploratory without a confirmatory p-value.

## Reporting commitments

- Existing `k=2` results are reported regardless of direction.
- Canonical-versus-legacy BI rank correlations are reported for every model.
- Exact BI overlaps and Jaccard similarities are reported.
- Positive language is limited to the observed conditional permutation family and effect size.
- Failure to reject is reported as “no detectable advantage under this design,” not equivalence or proof that BI does not work.
- The amendment, protocol JSON, execution manifest, and analysis code are frozen in the same public git commit before the first `k=4` run. That commit SHA is the analysis-code provenance anchor.
