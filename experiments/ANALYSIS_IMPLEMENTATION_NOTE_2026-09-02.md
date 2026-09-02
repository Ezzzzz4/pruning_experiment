# Analysis implementation note — 2026-09-02

The first execution of the frozen analysis began only after all 133 manifest configurations had completed. It stopped before writing a summary because `load_task_samples` treated `doc_hash` as a unique row identifier. Lambada OpenAI contains repeated content at distinct dataset rows: document IDs 1753 and 1889 share one hash, and IDs 3469 and 3624 share another.

The pairing key now requires the composite `(doc_id, doc_hash)`. This distinguishes repeated content while still detecting a dataset-order or content mismatch between configurations. All 25 official full-task records contain 19,534 unique composite identities, with identical identity sets and no hash mismatch for a fixed task and document ID.

This correction changes neither an outcome value nor any preregistered statistic, comparison, tie rule, seed, or confidence-interval procedure. The failed analysis attempt produced no result artifact because output is written only after every analysis stage succeeds.

The final report also materializes preregistered outputs that the frozen script left implicit: exploratory `k=2`, descriptive `k=8` dose response, edge-contact distributions, model-specific effect summaries, exact sample-count checks, and input provenance. These additions do not alter the primary `k=4` test.
