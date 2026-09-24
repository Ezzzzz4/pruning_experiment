# Post-evaluation verification and methodological clarifications — September 24, 2026

I re-examined the implementation and recorded results of the completed study at commit `1b33efbdf003e21841bcd24fbc007f920c7f4b4b`. This note records the checks and subsequent corrections. I retained the frozen experimental design and original observations. The verification described here was performed after evaluation.

## Numerical evidence

I checked the original JSONL logs using a separate script that did not import `experiments/statistics.py`. The checks covered:

- 133 unique successful configurations in manifest order: 25 full-task and 108 WikiText-only;
- all 133 original-record hashes and 133 sample-log hashes against the September 2 summary;
- 495,046 sample rows, with matching task counts and composite `(doc_id, doc_hash)` identities;
- reconstruction of 133 WikiText perplexities and 125 accuracy values from individual outcomes;
- all 75 paired McNemar comparisons and their within-family Holm corrections;
- frozen permutation bijections, transformed layer sets, and canonical/legacy Spearman correlations.

| Quantity | Recomputed value |
|---|---:|
| BI mean log degradation, primary `k=4` | 0.3739508014447414 |
| Lowest random mean log degradation | 0.7022065843813233 |
| Primary rank-tail value | 1/21 = 0.047619047619 |
| Edge-free control count | 8 |
| Edge-free rank-tail value | 1/9 = 0.111111111111 |
| Median-random minus BI, log scale | 1.4415176582861995 |
| Paired 10,000-resample document interval | [1.3778327439872184, 1.5052558076507556] |
| Exponentiated median-log contrast | 4.227106254173 |

I retained the [September 2 summary](results/confirmatory/summary.json) and [official aggregate export](results/confirmatory/official_runs.jsonl) unchanged. I extracted the text-free [evidence bundle](results/confirmatory/evidence/) from the original logs after checking their hashes. The bundle contains the numerical inputs needed to repeat the statistical calculations from a clone. This reanalysis checks recorded outcomes; it does not independently verify the original GPU computation or replace inference with the models and datasets.

## Interpretation corrections

### Permutation null hypothesis

I interpret the primary formula, `(1 + number of random objects no worse than BI) / 21`, as a finite-sample Monte Carlo rank test under a null of layer-label invariance. Conditional on overlap structure, BI's placement in the selection orbit is assumed exchangeable with uniform relabelings relative to pruning quality. Preserving overlaps and adding the identity do not establish this assumption. [Hemerik and Goeman, Definition 1 and Theorem 2](https://arxiv.org/html/1411.7565) state the invariance requirement.

All three primary BI sets equal `[14, 15, 16, 17]`. The primary orbit therefore consists of shared four-of-28 selections. The twenty controls represent twenty sampled sets rather than sixty independent observations. The reported p-value is neither an exhaustive tail calculation over 20,475 sets nor a test of equality of mean quality. The observed result, lower perplexity for BI than for all twenty controls, can also be reported descriptively.

The eight edge-free controls leave the diagnostic underpowered. I did not include a depth-matched or contiguous-middle-layer control. The comparison therefore cannot distinguish the effect of detailed BI ranking from avoidance of sensitive layer depths.

### Calibration sample and BI implementation comparison

I sampled 128 non-empty WikiText-2 training rows, of which 30 are headings, using [prepare_calibration.py](experiments/prepare_calibration.py). The texts and source-row identifiers remain in the committed corpus. The sampling procedure did not use the evaluation split. Equal-example weighting gives a heading the same weight as a long row, which limits the representativeness of the calibration sample.

Canonical and legacy pipelines differ in text length (256 versus 128 tokens), batch size (2 versus 8), precision, masking, flattening, and padding. I therefore report their rank correlations as a comparison of two complete pipelines. These correlations cannot isolate an effect of fp16 or padding, or establish an error in ShortGPT. I retained the frozen corpus and BI selections.

The bootstrap jointly resamples the same 62 evaluation documents across candidates and checkpoints. Its interval conditions on the fixed control panel and BI profiles; it omits uncertainty from different layer selections or calibration samples. `exp(median(log contrast))` gives a geometric midpoint at even sample size, rather than the arithmetic median of the exponentiated values. Lambada is last-word prediction, not a multiple-choice task.

## Engineering findings and repairs

| Finding on the reviewed snapshot | Repair | Effect on recorded evidence |
|---|---|---|
| Pruning left Qwen cache indices and configuration metadata stale; tests did not reuse the cache. | Qwen metadata is synchronized after pruning. Tests cover cached continuation after removing first, middle, and last layers. | All 133 recorded runs used `use_cache=False`; their likelihood results do not rely on cached continuation. |
| Resume trusted only a matching string run key. | Successful records are checked against the frozen manifest before skipping. Incompatible official settings are rejected before model loading. | Every completed record matched the original manifest. |
| The exported handler factory referenced missing ResNet/YOLO modules; docstrings described absent behavior. | Unsupported routes and their documentation were removed. | The experiment used the Qwen handler directly. |
| BI could return NaN silently; multiplication by a zero mask did not exclude NaN padding. | Nonfinite BI is rejected and padded distances are masked explicitly, with regression tests. | Every published canonical/legacy score is finite and matches its committed bundle. |
| Public files did not include inputs for sample-level reanalysis. | Aligned numerical sufficient statistics and a CPU-only verifier were added. Figures now read the tracked export. | Recorded scores and the historical summary are unchanged. |
| Frozen text hashes were computed with Windows CRLF endings; Git also produces LF checkouts. | Frozen text checks accept equivalent LF/CRLF byte representations. Raw log and binary hashes remain byte-strict. | The original digests are retained; changed JSON values are rejected. |

## Execution chronology and deviations

Times below are UTC, from local git and recorded run timestamps.

| Event | Timestamp / evidence |
|---|---|
| Freeze commit `134024e` | August 17, 19:28:16; August 18 in the project's UTC+5 time zone |
| First primary run | August 17, 23:29:27, `base:bi:k4:seednone`, running the freeze SHA |
| Last grid completion | September 1, 23:24:02 |
| Final analysis code and results publication commits | September 2, `8b25654` and `1b33efb` |
| This review and portable evidence | September 24 |

The protocol and manifest bytes have not changed since the freeze. I verified the local commit and run sequence but could not independently establish when GitHub received the original push. The available record supports a design recorded locally before evaluation, rather than independently authenticated public timestamping.

I extended execution beyond the original August 24 target after interruptions. All twenty primary objects completed in the frozen order. I retained the fixed sample size and report the complete grid; execution did not stop in response to the observed effect.

Nine indexed failures and five unindexed sample artifacts remain in the provenance report. The complete orphan baseline has identical per-item outcomes to the accepted baseline; its aggregate serialization failure is documented in commit `04b1ffc`. Partial failed instruction-model seed-7 outcomes also match. Two seed-13 partial logs differ from the accepted run on one finite likelihood each, showing that replay was not bit-for-bit deterministic. Their later nonfinite outputs triggered failure.

I retained model revisions, tasks, layers, seeds, dtype, harness revision, and context length across retries. During recovery, I changed code and some package revisions, departing from the original same-software retry rule. These deviations are documented in the [August 31](experiments/PROTOCOL_IMPLEMENTATION_NOTE_2026-08-31.md) and [September 2](experiments/ANALYSIS_IMPLEMENTATION_NOTE_2026-09-02.md) notes. The checks identified no missing or duplicate successful configuration or selective layer replacement.

## Remaining limits

I did not record immutable revision IDs or fingerprints for the evaluation datasets. The existing hashes establish within-study sample alignment, not a recoverable original dataset snapshot. With one checkpoint per training regime, I treat differences across checkpoints as descriptive. All tasks used the same untemplated likelihood setup, which does not measure each model's best prompted behavior. I did not measure speed, energy savings, or deployed generation quality.

All successful aggregate PPL values in this study are finite. The historical analysis takes logs of those aggregates. Extending it to runs whose exponentiated PPL overflows would require computing log-PPL directly from likelihood sums to avoid artificial ties between distinct catastrophic outcomes; the current dataset does not exercise that case.

On September 24, Windows application control blocked a DLL in the historical CUDA PyTorch environment (`WinError 4551`), and `nvidia-smi` was unavailable without elevated access. I ran code checks in a separate CPU environment with the same PyTorch release, leaving the historical environment and operating-system settings unchanged. I did not repeat 7B model inference on the GPU.

## Verification of the revised repository

- **125 tests passed** in an exported source tree with LF line endings, using Python 3.11, PyTorch `2.11.0+cpu`, and Transformers `5.15.0`. Model tests ran offline with tiny random Qwen configurations; they downloaded no checkpoints.
- Official preflight accepted every one of the 133 frozen manifest configurations. Regression tests reject altered BI scores, unsupported settings, missing provenance, and incompatible completed runs.
- A separate analysis environment had no PyTorch installed and the exported tree had no `results/lm_eval/` directory. It reproduced all nine published scientific sections and generated all five figures.
- The original protocol, manifest, calibration corpus, BI bundles, analysis script, summary, and aggregate export were preserved. Compile checks and dependency compatibility checks passed.

The [CI workflow](.github/workflows/verify.yml) runs the public-evidence replay, figure generation, and CPU tests on Ubuntu for subsequent pushes. Local LF testing was performed on Windows; CI provides the separate operating-system check.
