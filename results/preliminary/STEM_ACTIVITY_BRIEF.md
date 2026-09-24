# Abstract

I evaluated whether Block Influence (BI), introduced in ShortGPT, selects transformer blocks whose removal causes less performance degradation than random selection. The study compared three existing Qwen2.5-7B checkpoints on one laptop GPU using a pinned evaluation harness. After withdrawing conclusions from an earlier implementation, I conducted an exploratory two-layer pilot. WikiText perplexity ranged from 13.36 to 20.93 across its random selections. I subsequently fixed twenty new random layer-label permutations and the analysis rules before evaluating the primary four-layer comparison. The completed grid contains 133 configurations, including exploratory and secondary analyses.

At four removed layers, BI produced lower WikiText perplexity than all twenty random controls on each checkpoint. The Monte Carlo rank p-value was 0.0476 under the specified layer-label exchangeability null. BI also produced lower perplexity than all eight edge-free controls; the corresponding diagnostic p-value was 0.111. I interpret these findings within the evaluated models and sampled selections. The design does not establish superiority over a depth-matched pruning heuristic or a causal effect of fine-tuning.

Methods, results, and supporting materials are listed in the [research summary](../../PROJECT_CARD.md).
