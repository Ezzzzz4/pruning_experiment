# Portable numerical evidence

I provide the numerical inputs for reanalysis in `sufficient_statistics.json.gz`, approximately 0.97 MB compressed. The adjacent `.sha256` file contains the bundle's checksum.

## Contents

- Shared `(doc_id, doc_hash)` tables for each evaluation task; repeated text at different row IDs stays distinguishable.
- All 133 × 62 WikiText document log-likelihood and word-count pairs.
- Binary correctness vectors for all five accuracy tasks in the 25 full-task runs, including the exploratory pilot.
- Original run identifiers and source-log hashes, plus semantic hashes binding the tracked records, summary, manifest, and protocol.

The bundle excludes evaluation text, answers, prompts, model responses, and weights. The exporter checks local raw-log hashes before extracting numerical fields. Source bindings use semantic JSON hashes. The older BI-file digests allow only LF/CRLF conversion to match their historical Windows representation; changed scores still fail. Raw log and gzip hashes remain byte-strict.

## Verify

From the repository root in an environment installed from `requirements-analysis.txt`:

```text
python -m experiments.evidence_bundle verify
```

The command checks the checksum, identities, manifest coverage, task counts, every reconstructed PPL and accuracy, and then replays the primary and secondary statistics against `../summary.json`. It also recomputes BI selections, overlap, and canonical/legacy correlations from the committed BI bundles. It requires neither GPU nor PyTorch and writes no replacement summary.

Regenerating this export requires the original local `results/lm_eval/` logs. `python -m experiments.evidence_bundle export` refuses an existing output unless `--force` is supplied. The `verify` command uses the committed bundle and does not require those logs.

This is a record of measured outcomes, not a substitute for evaluating the model again. Published hashes link it to the original logs; they do not independently authenticate the original computation or provide a snapshot of the evaluation datasets.
