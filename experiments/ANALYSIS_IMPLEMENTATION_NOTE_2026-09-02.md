# Analysis implementation note — 2026-09-02

I first executed the frozen analysis after all 133 manifest configurations had completed. Execution stopped before writing a summary because `load_task_samples` treated `doc_hash` as a unique row identifier. Lambada OpenAI contains repeated content at distinct dataset rows: document IDs 1753 and 1889 share one hash, and IDs 3469 and 3624 share another.

I changed the pairing key to the composite `(doc_id, doc_hash)`. This distinguishes repeated content while retaining detection of dataset-order or content mismatches between configurations. All 25 official full-task records contain 19,534 unique composite identities, with identical identity sets and no hash mismatch for a fixed task and document ID.

This correction changes neither an outcome value nor any predeclared statistic, comparison, tie rule, seed, or confidence-interval procedure. The failed analysis attempt produced no result artifact because output is written only after every analysis stage succeeds.

I also added explicit outputs for analyses and checks specified in the protocol but left implicit in the frozen script: exploratory `k=2`, descriptive `k=8` dose response, edge-contact distributions, model-specific effect summaries, exact sample-count checks, and input provenance. These additions do not alter the primary `k=4` test.
