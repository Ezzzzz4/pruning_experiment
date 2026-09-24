# Confirmatory results

## Scope

I evaluated all 133 configurations in `experiments/experiment_manifest.json`. The primary comparison concerns WikiText word perplexity after removing four of 28 blocks. BI selected the same `[14, 15, 16, 17]` blocks in all three checkpoints. Each of twenty random controls is a shared four-block selection, evaluated on all three checkpoints. These are twenty selection objects rather than sixty independent replications.

I interpret the p-value under a null in which, conditional on the selection-overlap structure, BI's position in the permutation orbit is exchangeable with uniformly relabeled positions relative to pruning quality. The permutation construction does not establish this assumption. The observed ranking can also be reported descriptively. [Hemerik and Goeman](https://arxiv.org/html/1411.7565) give the invariance requirement for this form of random permutation test. In the [September 24 review](../../RESEARCH_REVIEW_2026-09-24.md), I made the previously implicit assumption explicit without changing the frozen statistic or outcomes.

## Primary result: k=4 WikiText

The statistic is the mean across checkpoints of `log(PPL_pruned / PPL_baseline)`. Lower is better.

| Checkpoint | Baseline PPL | BI PPL | BI / baseline | Random median PPL | Random range | BI rank |
|---|---:|---:|---:|---:|---:|---:|
| Base | 9.4967 | 12.6763 | 1.335x | 24.2469 | 16.4578–260421.3424 | 1/21 |
| Instruct | 10.1367 | 13.6588 | 1.347x | 28.7813 | 18.4209–361556.7230 | 1/21 |
| Math | 144.1987 | 246.1740 | 1.707x | 3589.8709 | 333.8323–660570.9400 | 1/21 |

BI's aggregate statistic is `0.3739508014`; all twenty controls are worse. The predeclared one-sided Monte Carlo rank p-value under the stated null is `1/21 = 0.047619`. It is not the exhaustive tail proportion among all 20,475 four-block subsets. The mean-within-model-rank robustness statistic also places BI first and returns the same p-value; it uses the same observations and is not an independent replication.

The paired 10,000-replicate document bootstrap estimates `median(random) - BI` as `1.4415176583` log-PPL units with a 95% interval of `[1.3778327440, 1.5052558077]`. Exponentiating this median-log contrast gives `4.227x`, with interval `[3.966x, 4.505x]`. At even sample size this is the geometric midpoint of the two middle multiplicative contrasts, not their arithmetic median. The interval resamples the 62 WikiText documents jointly while conditioning on the twenty controls, three checkpoints, and BI profiles; it excludes layer-panel and calibration-sample uncertainty.

## Edge-layer diagnostic

I fixed the edge set as `{0, 1, 26, 27}` before primary evaluation. Eight of twenty `k=4` permutation objects avoid every edge across all three checkpoints. BI produced lower perplexity than all eight; the corresponding conditional rank p-value is `1/9 = 0.111111`. Median aggregate degradation factors are:

- BI: `1.453x` baseline;
- edge-free random controls: `3.167x` baseline;
- edge-touching random controls: `37.658x` baseline.

The direction of the observed difference is unchanged in the edge-free subset. The comparison does not establish an advantage within this subset at the 0.05 level.

## Secondary full-task results

I evaluated random seeds 3–7 on all five accuracy tasks: four multiple-choice tasks and Lambada last-word prediction. BI exceeds their median accuracy for every model-task pair. BI ranks first of six candidates on ARC-Challenge, PIQA, HellaSwag, and Lambada, and third of six on Winogrande for all three checkpoints. I report the McNemar and bootstrap outputs in `summary.json` as comparisons of fixed selections rather than the primary strategy-level test.

## Descriptive k=8 dose response

| Checkpoint | BI PPL | BI / baseline | Random median PPL | Random range | BI rank |
|---|---:|---:|---:|---:|---:|
| Base | 21.8502 | 2.301x | 4824.1872 | 60.3057–77267108.4069 | 1/21 |
| Instruct | 26.4868 | 2.613x | 96119.2841 | 68.9802–123539512.4480 | 1/21 |
| Math | 731.8055 | 5.075x | 240112.3348 | 3669.6341–1049425.7874 | 1/21 |

BI ranks first in the frozen family at `k=8`. I treat this pruning level as a secondary descriptive analysis. The math-tuned checkpoint has a larger BI-to-baseline perplexity ratio than Base or Instruct. With no independent checkpoints within each training regime, I cannot attribute this difference to fine-tuning or establish its statistical significance.

## BI-definition sensitivity

Canonical-versus-legacy layer-rank Spearman correlations are `0.2899` (Base), `0.2978` (Instruct), and `0.6311` (Math). The two pipelines change padding, precision, masking, aggregation, batch size, and maximum input length (256 versus 128) together. Their different rankings do not identify which change caused the difference. Calibration uses 128 non-empty training rows, including 30 headings, equally weighted per row; these are not 128 full documents.

## Integrity and limitations

- The analysis requires exactly 133 unique successful manifest keys and exact agreement on model revisions, harness SHA, task lists, pruning indices, dtype, batch size, and sample counts.
- Every successful run contains the same 62 aligned WikiText documents. Aggregate PPL values were independently reconstructed from document log likelihoods.
- I retained nine failed official attempts in the provenance report. Retries of the same frozen experimental configurations later succeeded; I did not change or retry layer selections because of metric direction.
- Five unindexed local sample logs are listed separately in `summary.json`; they are interrupted, auxiliary, or orphan artifacts outside the 133-record evidence index.
- Successful execution spans several code SHAs and package revisions after documentation, serialization, and environment-recovery changes. Model revisions, harness revision, evaluation settings, and frozen layer selections remained fixed; the input inventory records each SHA. These are [disclosed technical deviations](../../RESEARCH_REVIEW_2026-09-24.md#execution-chronology-and-deviations) from the original same-software retry rule.
- The primary p-value is at the design's resolution limit. One control with an equal or lower primary statistic than BI would produce at least `2/21 = 0.095238`.
- Raw sample logs total about 1.7 GB and are not tracked. Their hashes remain published. The [compact text-free evidence](evidence/) now allows a fresh clone to recompute the numerical analysis without the raw texts or a GPU; it does not rerun model inference.
- Evaluation task names and harness code are frozen, but immutable Hugging Face dataset revisions or fingerprints were not recorded. Document hashes verify cross-run alignment here; they are not a full dataset snapshot.
- Generalization beyond these three related Qwen2.5-7B checkpoints requires new experiments.
- All evaluations used a 2,048-token context limit and no chat template. Performance differences do not measure each checkpoint's best prompted use.
- The study has no depth-matched pruning control and no inference-speed or energy benchmark. The `4.227x` contrast concerns perplexity degradation, not acceleration.
