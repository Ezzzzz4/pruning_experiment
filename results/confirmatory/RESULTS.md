# Confirmatory results

## Scope

All 133 configurations in `experiments/experiment_manifest.json` completed successfully. The primary inference concerns WikiText word perplexity after removing four of 28 blocks. It is conditional on the frozen family containing BI and twenty layer-label permutations; it is not a claim over every possible random-pruning distribution.

## Primary result: k=4 WikiText

The statistic is the mean across checkpoints of `log(PPL_pruned / PPL_baseline)`. Lower is better.

| Checkpoint | Baseline PPL | BI PPL | BI / baseline | Random median PPL | Random range | BI rank |
|---|---:|---:|---:|---:|---:|---:|
| Base | 9.4967 | 12.6763 | 1.335x | 24.2469 | 16.4578–260421.3424 | 1/21 |
| Instruct | 10.1367 | 13.6588 | 1.347x | 28.7813 | 18.4209–361556.7230 | 1/21 |
| Math | 144.1987 | 246.1740 | 1.707x | 3589.8709 | 333.8323–660570.9400 | 1/21 |

BI's aggregate statistic is `0.3739508014`; all twenty controls are worse. The preregistered one-sided exact p-value is therefore `1/21 = 0.047619`. The mean-within-model-rank robustness statistic also places BI first and returns the same p-value.

The paired 10,000-replicate document bootstrap estimates `median(random) - BI` as `1.4415176583` log-PPL units with a 95% interval of `[1.3778327440, 1.5052558077]`. On the multiplicative scale, the median random degradation is `4.227x` BI's, with interval `[3.966x, 4.505x]`. This interval measures uncertainty from the fixed 62 WikiText documents; it does not replace variation across layer selections.

## Edge-layer diagnostic

The edge set was frozen as `{0, 1, 26, 27}`. Eight of twenty `k=4` permutation objects avoid every edge across all three checkpoints. BI beats all eight, but the corresponding exact p-value is only `1/9 = 0.111111`. Median aggregate degradation factors are:

- BI: `1.453x` baseline;
- edge-free random controls: `3.167x` baseline;
- edge-touching random controls: `37.658x` baseline.

The observed advantage is not erased by removing edge-touching controls, but this experiment does not establish a beyond-edge advantage at the 0.05 level.

## Secondary full-task results

Only random seeds 3–7 ran all five multiple-choice tasks. BI exceeds their median accuracy for every model-task pair. BI ranks first of six candidates on ARC-Challenge, PIQA, HellaSwag, and Lambada, but third of six on Winogrande for all three checkpoints. McNemar and bootstrap outputs are retained in `summary.json`; they describe the fixed comparisons and are not the primary strategy-level test.

## Descriptive k=8 dose response

| Checkpoint | BI PPL | BI / baseline | Random median PPL | Random range | BI rank |
|---|---:|---:|---:|---:|---:|
| Base | 21.8502 | 2.301x | 4824.1872 | 60.3057–77267108.4069 | 1/21 |
| Instruct | 26.4868 | 2.613x | 96119.2841 | 68.9802–123539512.4480 | 1/21 |
| Math | 731.8055 | 5.075x | 240112.3348 | 3669.6341–1049425.7874 | 1/21 |

BI again ranks first in the frozen family, but `k=8` is a secondary descriptive analysis. The math-tuned checkpoint degrades more under BI than Base or Instruct; the design does not contain independent checkpoints within each training regime, so this difference cannot support a causal or significance claim about fine-tuning.

## BI-definition sensitivity

Canonical-versus-legacy layer-rank Spearman correlations are `0.2899` (Base), `0.2978` (Instruct), and `0.6311` (Math). Padding, precision, masking, and aggregation choices materially affect the BI ordering, especially for Base and Instruct.

## Integrity and limitations

- The analysis requires exactly 133 unique successful manifest keys and exact agreement on model revisions, harness SHA, task lists, pruning indices, dtype, batch size, and sample counts.
- Every successful run contains the same 62 aligned WikiText documents. Aggregate PPL values were independently reconstructed from document log likelihoods.
- Nine failed official attempts remain in the provenance report. Exact retries later succeeded; no configuration was changed or retried because of metric direction.
- Five unindexed local sample logs are listed separately in `summary.json`; they are interrupted, auxiliary, or orphan artifacts outside the 133-record evidence index.
- Successful execution spans several code SHAs due documentation and strict-JSON recovery changes. Model revisions, harness revision, evaluation settings, and frozen layer selections remain fixed; the input inventory records every SHA and artifact hash.
- The primary p-value is at the design's resolution limit. One control tying or beating BI would produce at least `2/21 = 0.095238`.
- Raw sample logs total about 1.7 GB and are not tracked. Their hashes are published, but a fresh clone cannot recompute the bootstrap or McNemar appendix without a separately supplied matching copy.
- Evaluation task names and harness code are frozen, but immutable Hugging Face dataset revisions or fingerprints were not recorded. Document hashes verify cross-run alignment here; they are not a full dataset snapshot.
- Generalization beyond these three related Qwen2.5-7B checkpoints requires new experiments.
