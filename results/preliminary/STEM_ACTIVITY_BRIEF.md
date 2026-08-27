# STEM activity brief

## Project in one sentence

I built and audited a reproducible experiment to test whether Block Influence, a transformer-layer pruning metric, selects removable layers more reliably than random selection and whether fine-tuning changes that reliability.

## What I did

I began with a pruning repository whose documentation overstated what its code and old benchmarks could support. I treated two independent audits as engineering requirements, withdrew the old conclusions, archived the earlier evidence, and rebuilt the experiment around fixed model and evaluation revisions.

The rebuilt pipeline loads 7-billion-parameter Qwen2.5 checkpoints on one laptop GPU, removes specified transformer blocks, evaluates six public language tasks through `lm-evaluation-harness`, logs per-example outputs, and records model, code, dependency, timing, and hardware provenance. I also implemented two Block Influence calculations so the current token-masked FP32 definition could be compared with the legacy FP16 calculation.

The first complete pilot compared an unpruned Qwen2.5-7B model with BI pruning and three random layer selections at two removed blocks. All five configurations covered 19,534 samples. The random selections produced WikiText perplexities from 13.36 to 20.93, showing that the choice of random layers created more variation than a three-seed control could characterize reliably.

## What changed because of the evidence

Instead of presenting the most favorable pilot comparison, I labeled the data exploratory and redesigned the confirmatory study before collecting its primary results. The new public protocol freezes twenty conditional random permutations, exact layer indices, model revisions, metric directions, failure rules, and analysis code. This makes a negative result—BI being indistinguishable from random selection—as reportable as a positive one.

## Current result and limit

The finished artifact is a reproducible exploratory pilot and a preregistered confirmatory design. The larger `k=4` experiment is not complete, so I do not claim that BI works, fails, or changes with fine-tuning. The project demonstrates experimental design, GPU systems debugging, statistical reasoning, reproducibility, and the decision to narrow a claim when the available evidence is insufficient.
