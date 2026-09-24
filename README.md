# Block Influence vs. Random Layer Pruning

I evaluated whether Block Influence (BI), introduced in [ShortGPT](https://arxiv.org/abs/2403.03853) (Men et al., 2024), selects transformer blocks whose removal causes less performance degradation than random selection. I compared three existing Qwen2.5-7B checkpoints without further training or fine-tuning.

[Research summary](PROJECT_CARD.md) · [Results](results/confirmatory/RESULTS.md) · [Verification and methodological clarifications](RESEARCH_REVIEW_2026-09-24.md).

## Status

Evaluation and analysis are complete for all 133 configurations in the frozen manifest.

I withdrew the earlier conclusions after an engineering audit identified problems in the implementation and evaluation methods. I retained the earlier JSON files and figures under [`results/archive/`](results/archive/) as a historical record and excluded them from the current analysis.

I report the exploratory `k=2` pilot, the frozen `k=4` comparison, and the predeclared descriptive `k=8` dose response across three Qwen2.5-7B checkpoints. BI selected the same four blocks, `[14, 15, 16, 17]`, in all three models. Their removal produced lower WikiText perplexity than each of twenty sampled random four-block selections on every checkpoint. The statistical interpretation below requires an explicit permutation-invariance null.

- [Exploratory pilot results](results/preliminary/PILOT_RESULTS_2026-08-27.md)
- [Exact pilot metrics and run identifiers](results/preliminary/pilot_k2_summary.json)
- [Confirmatory analysis summary](results/confirmatory/summary.json)
- [Readable confirmatory results](results/confirmatory/RESULTS.md)
- [Exported official aggregate records](results/confirmatory/official_runs.jsonl)
- [Confirmatory figures](results/confirmatory/figures/)
- [Research abstract](results/preliminary/STEM_ACTIVITY_BRIEF.md)
- [Frozen protocol amendment](experiments/PROTOCOL_AMENDMENT_2026-08-18.md)
- [Post-run analysis implementation note](experiments/ANALYSIS_IMPLEMENTATION_NOTE_2026-09-02.md)
- [Exact layer permutations and BI selections](experiments/permutation_protocol.json)
- [Resumable experiment manifest](experiments/experiment_manifest.json)

## Research question

The primary question is:

> At a fixed pruning level, does BI select blocks whose removal preserves model quality better than random block selection?

I also compared outcomes across three related 7B checkpoints. This comparison is descriptive because each training regime is represented by one checkpoint.

| Key | Model | Frozen Hugging Face revision |
|---|---|---|
| `base` | `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` |
| `instruct` | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| `math` | `Qwen/Qwen2.5-Math-7B-Instruct` | `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d` |

The design tests for a BI advantage; it does not test equivalence between the strategies.

## Revision of the initial study

I revised the implementation and evaluation procedure in response to the following audit findings:

| Earlier problem | Consequence | Revision |
|---|---|---|
| The README described classes and architecture that did not exist. | A reader could not map the claimed design to executable code. | This README describes only tracked files and tested entry points. |
| Bespoke language, vision, and GSM8K benchmarks used small prompt sets or proxy scores. | Those scores did not support a comparison of BI and random pruning on standard language-model evaluations. | I replaced the custom scoring with a pinned `lm-evaluation-harness` revision. |
| Two BI paths used different padding, precision, masking, and averaging rules. | The selected blocks could depend on implementation details instead of the metric's intended definition. | One canonical implementation uses token-wise FP32 cosine distance and masks padding. The legacy calculation remains only as a named sensitivity analysis. |
| Layer removal left stale metadata, mutated copied-handler state, replaced `Sequential` containers, and accepted invalid indices. | Later operations could observe a model structure different from the actual pruned model. | Targeted fixes preserve container types, validate indices, and keep handler metadata synchronized. |
| Old result files mixed strategies and lacked complete provenance. | Figures could combine incompatible runs, and results could not be reconstructed reliably. | Official runs append JSONL records with exact layers, seeds, model and code revisions, package versions, timings, and per-sample logs. |
| The pilot used three random layer selections. | The observed variation limited the precision of this comparison. | I fixed twenty conditional random permutations before primary evaluation. |

Git history preserves the old README. It is not part of the current scientific claim.

## Current experimental design

### Block Influence

For each decoder block, canonical BI computes the cosine distance between its input and output hidden states for each token. It then:

1. performs the cosine calculation in FP32;
2. excludes padding with `attention_mask`;
3. averages real tokens within each example;
4. averages examples, including a correctly weighted final partial batch;
5. removes all hooks and restores the model's previous train/eval state, even after failure.

I used a fixed calibration sample of 128 non-empty WikiText-2 training rows, including 30 heading rows, shared across all checkpoints. The sampling unit is a dataset row rather than a complete document. Equal weighting gives short headings the same weight as long passages. The stored BI bundles include the legacy pipeline and its rank correlation with the canonical pipeline. Canonical inputs are truncated at 256 tokens; legacy inputs at 128, so the comparison changes text context as well as numerical settings.

### Evaluation

The harness revision is fixed at `8a07e1110d060de48cfc7a9a7987b7659060b60b`. Every full evaluation uses the complete datasets and records sample-level outputs.

| Task | Reported metric | Direction |
|---|---|---|
| WikiText | word perplexity | lower is better |
| ARC-Challenge | normalized accuracy | higher is better |
| PIQA | normalized accuracy | higher is better |
| Winogrande | accuracy | higher is better |
| HellaSwag | normalized accuracy | higher is better |
| Lambada OpenAI | accuracy | higher is better |

WikiText perplexity at `k=4` is the primary endpoint. The five accuracy tasks are secondary outcomes: four are multiple choice, while Lambada is last-word prediction scored through likelihoods. All six tasks use likelihood evaluation without free-form generation. The evaluation context limit is 2,048 tokens, batch size is 4, and no chat template is applied, including for Instruct and Math.

### Random control

The confirmatory design applies twenty frozen permutations of the 28 layer labels to the BI-selected sets. Applying the same permutation across checkpoints preserves their observed overlap structure and keeps `k=4` nested inside `k=8`. At `k=4`, each control is one shared uniform four-layer set evaluated on three models, not three independent selections. Exact indices are stored in [`experiments/permutation_protocol.json`](experiments/permutation_protocol.json).

The test assumes that, under its null, BI's placement on physical layer positions is exchangeable with uniform relabelings relative to pruning quality, conditional on the overlap structure. Preserving overlaps and including the identity do not establish that assumption. Under this null, the identity-plus-twenty rank formula gives a finite-sample Monte Carlo test; it does not enumerate all 20,475 four-layer subsets or test equality of mean performance. See [Hemerik and Goeman](https://arxiv.org/html/1411.7565) and the [dated interpretation clarification](RESEARCH_REVIEW_2026-09-24.md).

The manifest contains 133 configurations: 25 full six-task runs and 108 WikiText-only runs. The completed evidence index contains one successful official record for every manifest key.

## Confirmatory result

The primary endpoint is WikiText word perplexity at `k=4`. Lower is better. The primary statistic is the mean across models of `log(PPL_pruned / PPL_baseline)`.

| Model | Baseline PPL | BI PPL | Random PPL median | Random PPL range |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 9.4967 | 12.6763 | 24.2469 | 16.4578-260421.3424 |
| Qwen2.5-7B-Instruct | 10.1367 | 13.6588 | 28.7813 | 18.4209-361556.7230 |
| Qwen2.5-Math-7B-Instruct | 144.1987 | 246.1740 | 3589.8709 | 333.8323-660570.9400 |

Primary permutation test:

- BI statistic: `0.3739508014`
- one-sided Monte Carlo rank p-value under the stated null: `1/21 = 0.047619`
- robustness statistic, mean within-model rank: BI rank `1.0`, p-value `0.047619`
- paired WikiText document bootstrap for median-random advantage over BI: `1.4415` log-PPL units, 95% CI `[1.3778, 1.5053]`

The exponentiated median-log contrast is `4.23x` (`95% CI [3.97x, 4.51x]`). This summarizes perplexity degradation rather than execution speed. The interval resamples documents while holding the twenty controls, three checkpoints, and BI profiles fixed. It excludes uncertainty from sampling different layer sets or calibration texts. The p-value is the smallest attainable under the 21-object design: one control with an equal or lower primary statistic than BI would have raised it to at least `2/21 = 0.095238`.

Only 8 of the 20 random permutation objects avoided layers `{0, 1, 26, 27}` across all three models. BI ranked better than every edge-free control, but the predeclared conditional rank p-value within that subset is `1/9 = 0.111111`. The primary test rejects its stated invariance null at 0.05; the edge-free diagnostic does not. There is also no depth-matched or contiguous-middle-block control, so the study does not isolate the benefit of detailed BI ranking from simple depth heuristics.

I treated the five full-task random seeds (`3-7`) as secondary, descriptive comparisons. BI exceeds the random median on all five tasks for every checkpoint. It ranks first of six on ARC-Challenge, PIQA, HellaSwag, and Lambada, and third of six on Winogrande for all three models. I report the McNemar comparisons in the appendix because they concern particular layer sets rather than a strategy-level effect across tasks.

The predeclared `k=8` dose response is descriptive rather than primary. BI ranks first of 21 for every checkpoint. Relative to each model's baseline, BI increases WikiText PPL by `2.30x` for Base, `2.61x` for Instruct, and `5.07x` for Math; the corresponding random medians are higher. The BI-to-baseline ratio is largest for the Math checkpoint, but one checkpoint per training regime cannot establish a statistical effect of fine-tuning.

![Primary k=4 permutation ranking](results/confirmatory/figures/primary_k4_permutation_ranking.png)

![WikiText dose response](results/confirmatory/figures/k8_dose_response.png)

Spearman correlations between the canonical and legacy layer rankings are:

| Model | Spearman correlation |
|---|---:|
| Qwen2.5-7B | 0.2899 |
| Qwen2.5-7B-Instruct | 0.2978 |
| Qwen2.5-Math-7B-Instruct | 0.6311 |

These are correlations between two complete pipelines. Precision, masking, flattening, padding, batch size, and maximum text length all change together. This is not an ablation identifying any one cause, nor a replication that establishes an error in ShortGPT's published results.

## Completed pilot

The published pilot compares one baseline, BI, and three random selections at `k=2` on Qwen2.5-7B. Each configuration covers 19,534 task or document samples.

| Selection | Removed blocks | WikiText PPL |
|---|---:|---:|
| Baseline | — | 9.4967 |
| BI | 16, 17 | 10.8339 |
| Random seed 0 | 3, 14 | 13.3581 |
| Random seed 1 | 23, 26 | 20.9333 |
| Random seed 2 | 3, 20 | 13.4949 |

BI had lower perplexity than these three random selections, but three controls cannot characterize the random-selection distribution. One random selection also exceeded BI on Winogrande. I therefore increased the number of random controls for the primary comparison and retained the pilot as exploratory evidence.

## Repository map

```text
src/
  core/block_influence.py         canonical and legacy BI calculations
  handlers/universal_handler.py  layer discovery and removal
experiments/
  benchmark.py                   one pinned lm-eval configuration
  run_grid.py                    resumable manifest executor
  prepare_calibration.py         fixed WikiText calibration sample
  prepare_protocol.py            frozen permutations and run manifest
  statistics.py                  implementation of the frozen analysis rules
  visualize.py                   figures for the new confirmatory schema
  evidence_bundle.py             text-free export and CPU-only result replay
  bi/                            stored BI bundles for three checkpoints
  calibration/                   shared calibration corpus
  experiment_manifest.json       exact 133-configuration queue
  permutation_protocol.json      exact model revisions and layer sets
results/
  preliminary/                   tracked exploratory pilot summary
  confirmatory/                  tracked results, aggregate records, and figures
  archive/                       withdrawn historical outputs
  lm_eval/                       local append-only raw runs and samples
tests/                            unit and integration tests
```

## Recheck the published results without a GPU

From a fresh clone, Python 3.11 and the small analysis environment suffice. No model weights, PyTorch, Hugging Face credentials, or original prompt texts are needed:

```powershell
py -3.11 -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv-analysis\Scripts\python.exe -m experiments.evidence_bundle verify
.\.venv-analysis\Scripts\python.exe -m experiments.visualize --output-dir .omx/reproduced-figures
```

On Linux/macOS, use `python3.11` and `.venv-analysis/bin/python`. The compact [evidence bundle](results/confirmatory/evidence/) contains aligned document identities and numerical outcomes. Verification reconstructs aggregate metrics, the primary comparison, bootstrap intervals, and McNemar outputs against the September 2 summary. This reproduces the analysis of recorded scores; rerunning model inference still requires the evaluation environment and datasets.

## Run code checks and model evaluations

The tested environment uses Python 3.11 on Windows and an NVIDIA RTX 5090 Laptop GPU. Dependencies, including the CUDA 12.8 PyTorch build and harness git revision, are pinned in [`requirements.txt`](requirements.txt).

The [automated verification workflow](.github/workflows/verify.yml) replays the public evidence before installing model dependencies, then runs the unit tests with CPU PyTorch and a tiny random Qwen model. It downloads no model checkpoints.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Inspect the frozen queue without starting evaluations:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_grid --official-run --dry-run
```

An official run requires a clean git worktree, the pinned harness revision, the pinned model revision, and CUDA. To resume the queue intentionally:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_grid --official-run
```

Raw official harness outputs live under `results/lm_eval/`, which is ignored by git because the sample logs total about 1.7 GB. Their SHA-256 hashes remain in the original summary. The public compact bundle makes statistical replay possible without those files; it does not contain prompts, model responses, or a dataset snapshot. With the raw artifacts present, a separate full audit can be generated without overwriting the historical report:

```powershell
.\.venv\Scripts\python.exe -m experiments.statistics --output .omx/raw-recheck/summary.json --records-output .omx/raw-recheck/official_runs.jsonl
```

## Interpretation and limitations

I observed lower WikiText perplexity for BI than for the twenty sampled controls at `k=4` on three related Qwen2.5-7B checkpoints. I limit this conclusion to the evaluated models and selections. The edge-free subset is underpowered, and differences across checkpoints remain descriptive in the absence of independent fine-tuning replicates. I did not measure inference speed, energy consumption, or deployment performance. Execution extended beyond the original August 24 target; the [review timeline](RESEARCH_REVIEW_2026-09-24.md) records the actual dates and post-run changes.

I retained nine failed official attempts in the provenance record. Technical fixes changed some code and dependency revisions between attempts; the layer selections and tasks stayed fixed. I pinned the harness revision but did not record immutable evaluation-dataset revisions or fingerprints. Per-document hashes verify within-study alignment rather than a complete dataset snapshot.

## References and software

- Men et al. (2024), [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853), introduced BI and BI-based block removal.
- Hemerik and Goeman (2018), [Exact testing with random permutations](https://arxiv.org/html/1411.7565), state the invariance assumptions for the random permutation test.
- The three pretrained checkpoints are from [Qwen](https://huggingface.co/Qwen). I used their frozen revisions listed above without additional training.
- Task definitions and scoring are from [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/8a07e1110d060de48cfc7a9a7987b7659060b60b). The tasks and datasets are third-party work.
- PyTorch, Transformers, NumPy, SciPy, and Matplotlib support model execution and analysis. Package versions are listed in [`requirements.txt`](requirements.txt) and [`requirements-analysis.txt`](requirements-analysis.txt).

## License

The repository code is distributed under the MIT License; see [`LICENSE`](LICENSE). Referenced models and datasets retain their own terms.
