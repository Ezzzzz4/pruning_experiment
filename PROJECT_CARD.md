# Block Influence versus Random Layer Selection in Qwen2.5-7B

## Objective

I evaluated whether Block Influence (BI), introduced in [ShortGPT](https://arxiv.org/abs/2403.03853), identifies transformer blocks whose removal causes less performance degradation than random selection. BI measures cosine distance between a block's input and output representations; blocks with lower scores are selected for removal.

## Methods

I evaluated three existing Qwen2.5-7B checkpoints, Base, Instruct, and Math-Instruct, on one RTX 5090 laptop GPU with 24 GB of memory. Evaluation used a pinned version of `lm-evaluation-harness`. Model weights were not trained or fine-tuned during the study, including after pruning.

After an audit identified limitations in my initial implementation and benchmarks, I withdrew the earlier conclusions and archived their supporting files. An exploratory two-block pilot then showed substantial variation between three random selections. Before the first four-block evaluation, I fixed twenty new random layer-label permutations and the analysis rules.

The completed grid contains 133 configurations: 25 evaluated across six tasks and 108 on WikiText alone. The primary endpoint is WikiText word perplexity after removing four of 28 blocks. Eight-block removal is a secondary descriptive analysis. Each random permutation is shared across the three checkpoints, preserving the overlap structure of the BI selections.

## Results

BI selected blocks 14–17 in all three checkpoints. Removing these blocks produced lower WikiText perplexity than all twenty random selections on each model. Lower perplexity indicates better prediction of the evaluation text.

| Checkpoint | Unpruned PPL | BI, four blocks removed | Median random PPL |
|---|---:|---:|---:|
| Base | 9.50 | 12.68 | 24.25 |
| Instruct | 10.14 | 13.66 | 28.78 |
| Math-Instruct | 144.20 | 246.17 | 3,589.87 |

The predeclared Monte Carlo rank test gives p = 0.0476 under the stated layer-label exchangeability null. Eight controls avoided the edge layers. BI had lower perplexity than all eight, but the corresponding diagnostic gives p = 0.111. On Winogrande, two of five full-task random controls achieved higher accuracy than BI for every checkpoint.

## Limitations

I interpret the findings as a comparison within the specified models, evaluation setup, and sampled layer selections. The design does not establish a causal effect of fine-tuning or an advantage over a depth-matched pruning heuristic. Calibration uses a small sample that includes headings. The canonical-versus-legacy BI comparison changes several settings jointly and does not isolate their individual effects. Inference speed was not measured.

## Materials

- [Results and limits](results/confirmatory/RESULTS.md)
- [Frozen design](experiments/PROTOCOL_AMENDMENT_2026-08-18.md) and [exact selections](experiments/permutation_protocol.json)
- [Numerical verification and dated corrections](RESEARCH_REVIEW_2026-09-24.md)
- [CPU-only reproduction instructions](README.md#recheck-the-published-results-without-a-gpu)
- [Original aggregate records](results/confirmatory/official_runs.jsonl) and [compact numerical evidence](results/confirmatory/evidence/)
