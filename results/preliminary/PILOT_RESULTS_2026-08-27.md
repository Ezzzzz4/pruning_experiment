# Exploratory pilot snapshot — 2026-08-27

## Status

The repository contains a complete exploratory comparison for Qwen2.5-7B at `k=2`: one unpruned baseline, one Block Influence (BI) selection, and three seeded random selections. Every run evaluated the full six-task suite with `lm-evaluation-harness`, without sample limits. Each record contains 19,534 task or document samples.

This snapshot supports a project-progress report. It does **not** answer the confirmatory research question. The frozen `k=4` permutation study is incomplete.

Nine successful official records exist locally: all three model baselines; the five Qwen2.5-7B `k=2` records summarized here; and preliminary `k=4` BI records for the base and instruct checkpoints. Only the coherent `k=2` comparison is reported below.

## Exact pilot results

Higher is better for the five accuracy columns. Lower is better for WikiText word perplexity.

| Selection | Removed blocks | ARC-C acc_norm | PIQA acc_norm | Winogrande acc | HellaSwag acc_norm | Lambada acc | WikiText PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | — | 0.5102 | 0.7971 | 0.7285 | 0.7892 | 0.7200 | 9.4967 |
| BI | 16, 17 | 0.4650 | 0.7922 | 0.6567 | 0.7453 | 0.6412 | 10.8339 |
| Random seed 0 | 3, 14 | 0.4343 | 0.7622 | 0.5801 | 0.6680 | 0.5835 | 13.3581 |
| Random seed 1 | 23, 26 | 0.4309 | 0.7568 | 0.6961 | 0.7068 | 0.4320 | 20.9333 |
| Random seed 2 | 3, 20 | 0.4514 | 0.7514 | 0.5927 | 0.6729 | 0.5302 | 13.4949 |

BI produced lower perplexity and higher accuracy than all three random selections on four of the five accuracy tasks. One random selection exceeded BI on Winogrande. These comparisons are observations from a small exploratory sample, not significance-tested evidence that BI outperforms random selection.

The random controls varied substantially despite removing the same number of blocks. Their WikiText perplexities ranged from 13.36 to 20.93; their Winogrande accuracies ranged from 0.580 to 0.696. This instability motivated replacing the original `random×3` design with the frozen 20-permutation protocol.

## Reproducibility record

- Model: `Qwen/Qwen2.5-7B`
- Model revision: `d149729398750b98c0af14eb82c78cfe92750796`
- Harness revision: `8a07e1110d060de48cfc7a9a7987b7659060b60b`
- Tasks: ARC-Challenge, PIQA, Winogrande, HellaSwag, Lambada OpenAI, and WikiText
- Evaluation dtype: FP16
- Hardware: NVIDIA GeForce RTX 5090 Laptop GPU
- Exact metrics and run identifiers: [`pilot_k2_summary.json`](pilot_k2_summary.json)
- Frozen confirmatory protocol: [`../../experiments/PROTOCOL_AMENDMENT_2026-08-18.md`](../../experiments/PROTOCOL_AMENDMENT_2026-08-18.md)
- Frozen layer selections and model revisions: [`../../experiments/permutation_protocol.json`](../../experiments/permutation_protocol.json)

The baseline record uses code revision `04b1ffc9be4f68f599b1038c6a0955752e4f26d7`; the four pruned records use `50ae203dcd07fb729bc2aad5372b385671801996`. The benchmark implementation is identical between these revisions. The intervening commit adds the resumable grid runner and its tests.

## Limits

- The `k=2` data were inspected before the confirmatory protocol was frozen, so they remain exploratory.
- Three random selections cannot estimate the full random-selection distribution reliably.
- One checkpoint cannot test whether BI's predictive value depends on fine-tuning.
- The incomplete `k=4` results are omitted to avoid selective interpretation.

The valid conclusion is narrow: the rebuilt pipeline completed a full-task pruning pilot, and the observed sensitivity to random layer choice justified a stronger confirmatory design. No confirmatory claim about BI is made.
