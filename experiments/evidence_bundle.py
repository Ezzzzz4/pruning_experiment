"""Portable sufficient-statistics bundle for the confirmatory pruning analysis.

The bundle is a compact reanalysis artifact. It is not a model-inference log:
it stores only the item identities and metric values needed to replay the
published statistics without downloading models, importing torch, or keeping
the large lm-eval sample logs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments import statistics
from experiments.benchmark import decode_non_finite_float, json_safe
from experiments.frozen_files import require_frozen_text_hash


SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = Path("results/confirmatory/evidence/sufficient_statistics.json.gz")
DEFAULT_SUMMARY = Path("results/confirmatory/summary.json")
DEFAULT_OFFICIAL_RECORDS = Path("results/confirmatory/official_runs.jsonl")
DEFAULT_PROTOCOL = Path("experiments/permutation_protocol.json")
DEFAULT_MANIFEST = Path("experiments/experiment_manifest.json")
FULL_SAMPLE_TASKS = tuple(statistics.MC_PRIMARY_METRICS)
FLOAT_RTOL = 1e-12
FLOAT_ATOL = 1e-12
COMPARE_SECTIONS = (
    "primary_permutation",
    "paired_document_bootstrap",
    "exploratory_k2",
    "dose_response",
    "full_task_aggregate_summary",
    "mcnemar_appendix",
    "full_task_bootstrap_cis",
    "bi_overlap",
    "canonical_legacy_spearman",
)


def resolve_repo_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def public_path(path: Path | str) -> str:
    path = resolve_repo_path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def repo_json_path(path_value: str) -> Path:
    return resolve_repo_path(Path(path_value.replace("\\", "/")))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from exc
    return rows


def read_bundle(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_bundle(path: Path, bundle: dict[str, Any], *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing bundle: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(bundle)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz:
            gz.write(payload)
    path.with_name(f"{path.name}.sha256").write_text(
        sha256_file(path) + "  " + path.name + "\n", encoding="utf-8"
    )


def verify_bundle_sidecar(path: Path) -> str:
    sidecar = path.with_name(f"{path.name}.sha256")
    if not sidecar.is_file():
        raise ValueError(f"Missing evidence sidecar: {sidecar}")
    line = sidecar.read_text(encoding="utf-8").strip()
    parts = line.split()
    if len(parts) != 2 or parts[1] != path.name:
        raise ValueError(f"Invalid evidence sidecar format: {sidecar}")
    actual = sha256_file(path)
    if parts[0] != actual:
        raise ValueError(f"Evidence bundle SHA mismatch: {parts[0]} != {actual}")
    return actual


def load_official_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        run_key = record.get("provenance", {}).get("run_key")
        if not isinstance(run_key, str):
            raise ValueError(f"Missing provenance.run_key in {path}.")
        if run_key in records:
            raise ValueError(f"Duplicate official record for {run_key} in {path}.")
        records[run_key] = record
    return records


def sample_identity(sample: dict[str, Any], task: str, path: Path) -> list[Any]:
    doc_id = sample.get("doc_id")
    doc_hash = sample.get("doc_hash")
    if doc_id is None or not isinstance(doc_hash, str) or not doc_hash:
        raise ValueError(
            f"Missing stable {task} sample identity in {path}: "
            f"doc_id={doc_id!r}, doc_hash={doc_hash!r}."
        )
    return [doc_id, doc_hash]


def identity_key(identity: list[Any]) -> str:
    return json.dumps(identity, separators=(",", ":"), ensure_ascii=False)


def metric_key(task: str) -> str:
    return statistics.MC_PRIMARY_METRICS[task]


def collect_needed_samples(
    record: dict[str, Any], expected_tasks: Iterable[str]
) -> dict[str, dict[str, Any]]:
    path = resolve_repo_path(record["sample_log_path"])
    if not path.is_file():
        raise ValueError(f"Missing sample log for {record['provenance']['run_key']}: {path}")
    expected_tasks = set(expected_tasks)
    by_task: dict[str, dict[str, Any]] = {task: {} for task in expected_tasks}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") != record["run_id"]:
                raise ValueError(
                    f"Sample run_id mismatch for {record['provenance']['run_key']} "
                    f"at {path}:{line_number}."
                )
            task = row.get("task")
            if task not in expected_tasks:
                raise ValueError(f"Unexpected task {task!r} in {path}:{line_number}.")
            sample = row.get("sample")
            if not isinstance(sample, dict):
                raise ValueError(f"Invalid sample row in {path}:{line_number}.")
            identity = sample_identity(sample, task, path)
            key = identity_key(identity)
            if key in by_task[task]:
                raise ValueError(f"Duplicate {task} identity {key} in {path}.")
            if task == "wikitext":
                pair = sample.get("word_perplexity")
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(f"Invalid WikiText pair in {path}:{line_number}.")
                by_task[task][key] = [encode_number(pair[0]), encode_number(pair[1])]
            else:
                metric = statistics.MC_PRIMARY_METRICS.get(task)
                if metric is None or metric not in sample:
                    raise ValueError(f"Missing metric for {task} in {path}:{line_number}.")
                by_task[task][key] = encode_number(sample[metric])
    return by_task


def encode_number(value: Any) -> Any:
    numeric = float(decode_non_finite_float(value))
    if math.isnan(numeric):
        raise ValueError("NaN is not a valid evidence metric.")
    if numeric == math.inf:
        return {"__non_finite_float__": "positive_infinity"}
    if numeric == -math.inf:
        return {"__non_finite_float__": "negative_infinity"}
    return numeric


def decode_number(value: Any) -> float:
    return float(decode_non_finite_float(value))


def ordered_task_identities(
    task: str,
    run_samples_by_key: dict[str, dict[str, dict[str, Any]]],
    candidate_run_keys: Iterable[str],
) -> list[list[Any]]:
    reference: list[list[Any]] | None = None
    for run_key in candidate_run_keys:
        task_samples = run_samples_by_key[run_key].get(task)
        if task_samples is None:
            continue
        identities = [json.loads(key) for key in sorted(task_samples)]
        if reference is None:
            reference = identities
        elif identities != reference:
            raise ValueError(f"{task} item identities are not aligned for {run_key}.")
    if reference is None:
        raise ValueError(f"No samples found for task {task}.")
    return reference


def aggregate_from_binary(values: list[Any]) -> float:
    decoded = [decode_number(value) for value in values]
    if any(value not in {0.0, 1.0} for value in decoded):
        raise ValueError("Binary task evidence must contain only 0/1 metric values.")
    return float(np.mean(decoded))


def reconstruct_wikitext_ppl(values: list[list[Any]]) -> float:
    log_likelihood = [decode_number(pair[0]) for pair in values]
    words = [float(pair[1]) for pair in values]
    if any(math.isnan(value) or value == math.inf for value in log_likelihood):
        raise ValueError("Invalid WikiText log likelihood in evidence.")
    if any((not math.isfinite(value)) or value <= 0 for value in words):
        raise ValueError("Invalid WikiText word count in evidence.")
    return math.exp(-sum(log_likelihood) / sum(words))


def selected_bi_indices(bundle: dict[str, Any], k: int) -> list[int]:
    ordered = sorted(
        (int(index) for index in bundle["canonical"]),
        key=lambda index: (float(bundle["canonical"][str(index)]), index),
    )
    return sorted(ordered[:k])


def overlap_summary(sets_by_model: dict[str, list[int]]) -> dict[str, Any]:
    model_keys = sorted(sets_by_model)
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(model_keys):
        for right in model_keys[left_index + 1 :]:
            left_set = set(sets_by_model[left])
            right_set = set(sets_by_model[right])
            intersection = sorted(left_set & right_set)
            union = left_set | right_set
            pairwise[f"{left}__{right}"] = {
                "intersection": intersection,
                "intersection_size": len(intersection),
                "jaccard": len(intersection) / len(union),
            }
    common = sorted(set.intersection(*(set(sets_by_model[key]) for key in model_keys)))
    return {"pairwise": pairwise, "three_way_intersection": common}


def spearman_from_bi_bundle(bundle: dict[str, Any]) -> float:
    indices = sorted(int(index) for index in bundle["canonical"])
    canonical = np.array([float(bundle["canonical"][str(index)]) for index in indices])
    legacy = np.array([float(bundle["legacy"][str(index)]) for index in indices])
    canonical_ranks = statistics.rankdata(canonical, method="average")
    legacy_ranks = statistics.rankdata(legacy, method="average")
    return float(np.corrcoef(canonical_ranks, legacy_ranks)[0, 1])


def recompute_bi_sections(protocol: dict[str, Any]) -> dict[str, Any]:
    bundles = {}
    spearman = {}
    indices_by_k: dict[str, dict[str, list[int]]] = {"4": {}, "8": {}}
    for model_key in statistics.MODEL_KEYS:
        model_protocol = protocol["models"][model_key]
        path = repo_json_path(model_protocol["bi_path"])
        require_frozen_text_hash(path, model_protocol["bi_sha256"])
        bundle = read_json(path)
        bundles[model_key] = bundle
        spearman[model_key] = spearman_from_bi_bundle(bundle)
        for k in indices_by_k:
            selected = selected_bi_indices(bundle, int(k))
            if selected != model_protocol["bi_indices"][k]:
                raise ValueError(f"BI selected-index mismatch for {model_key}, k={k}.")
            indices_by_k[k][model_key] = selected
    return {
        "bi_overlap": {
            k: overlap_summary(indices_by_model)
            for k, indices_by_model in indices_by_k.items()
        },
        "canonical_legacy_spearman": spearman,
    }


def build_bundle(
    *,
    summary_path: Path,
    official_records_path: Path,
    protocol_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    protocol = read_json(protocol_path)
    manifest = read_json(manifest_path)
    official_rows = read_jsonl(official_records_path)
    records = load_official_records(official_records_path)
    inventory = summary["analysis_provenance"]["input_runs"]

    expected_keys = [config["run_key"] for config in manifest["configs"]]
    if sorted(records) != sorted(expected_keys):
        missing = sorted(set(expected_keys) - set(records))
        extra = sorted(set(records) - set(expected_keys))
        raise ValueError(f"Official record mismatch: missing={missing}, extra={extra}.")

    config_by_key = {config["run_key"]: config for config in manifest["configs"]}
    run_samples_by_key: dict[str, dict[str, dict[str, Any]]] = {}
    for run_key in expected_keys:
        record = records[run_key]
        config = config_by_key[run_key]
        expected_inventory = inventory.get(run_key)
        if expected_inventory is None:
            raise ValueError(f"Missing published inventory for {run_key}.")
        if record.get("run_id") != expected_inventory["run_id"]:
            raise ValueError(f"Official record run_id mismatch for {run_key}.")
        sample_path = resolve_repo_path(record["sample_log_path"])
        sample_sha = sha256_file(sample_path)
        if sample_sha != expected_inventory["sample_log_sha256"]:
            raise ValueError(f"Raw sample hash mismatch for {run_key}.")
        run_samples_by_key[run_key] = collect_needed_samples(record, config["tasks"])

    full_run_keys = [
        run_key
        for run_key in expected_keys
        if set(statistics.MC_PRIMARY_METRICS).issubset(run_samples_by_key[run_key])
    ]
    if len(full_run_keys) != 25:
        raise ValueError(f"Expected 25 full item-level runs, found {len(full_run_keys)}.")

    identities = {
        "wikitext": ordered_task_identities(
            "wikitext", run_samples_by_key, expected_keys
        ),
        **{
            task: ordered_task_identities(task, run_samples_by_key, full_run_keys)
            for task in FULL_SAMPLE_TASKS
        },
    }

    runs: dict[str, Any] = {}
    for run_key in expected_keys:
        record = records[run_key]
        sample_counts = inventory[run_key]["sample_counts"]
        run_entry: dict[str, Any] = {
            "run_id": record["run_id"],
            "sample_log_sha256": inventory[run_key]["sample_log_sha256"],
            "published_record_sha256": canonical_json_sha256(record),
            "source_record_sha256": inventory[run_key]["record_sha256"],
            "sample_counts": sample_counts,
            "wikitext": [],
        }
        wikitext_samples = run_samples_by_key[run_key].get("wikitext")
        if wikitext_samples is None:
            raise ValueError(f"Missing WikiText samples for {run_key}.")
        for identity in identities["wikitext"]:
            pair = wikitext_samples.get(identity_key(identity))
            if pair is None:
                raise ValueError(f"Missing WikiText item {identity} in {run_key}.")
            run_entry["wikitext"].append(pair)
        expected_ppl = inventory[run_key]["reconstructed_wikitext_word_perplexity"]
        actual_ppl = reconstruct_wikitext_ppl(run_entry["wikitext"])
        if not math.isclose(actual_ppl, expected_ppl, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"WikiText reconstruction mismatch for {run_key}.")

        tasks: dict[str, Any] = {}
        for task in FULL_SAMPLE_TASKS:
            task_samples = run_samples_by_key[run_key].get(task)
            if task_samples is None:
                continue
            values = []
            key = metric_key(task)
            for identity in identities[task]:
                value = task_samples.get(identity_key(identity))
                if value is None:
                    raise ValueError(f"Missing {task} item {identity} in {run_key}.")
                values.append(value)
            aggregate = aggregate_from_binary(values)
            expected = statistics.aggregate_metric(record, task, key)
            if not math.isclose(aggregate, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{task} aggregate mismatch for {run_key}.")
            tasks[task] = values
        if tasks:
            run_entry["tasks"] = tasks
        runs[run_key] = run_entry

    official_semantic = [
        json_safe(record)
        for record in (records[run_key] for run_key in expected_keys)
    ]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "confirmatory_sufficient_statistics",
        "description": (
            "Portable reanalysis bundle: enough item-level statistics to replay "
            "the published confirmatory analysis without model inference."
        ),
        "metadata": {
            "summary_path": public_path(summary_path),
            "summary_raw_sha256": sha256_file(summary_path),
            "summary_semantic_sha256": canonical_json_sha256(summary),
            "official_records_path": public_path(official_records_path),
            "official_records_raw_sha256": sha256_file(official_records_path),
            "official_records_semantic_sha256": canonical_json_sha256(official_semantic),
            "protocol_path": public_path(protocol_path),
            "protocol_raw_sha256": sha256_file(protocol_path),
            "protocol_semantic_sha256": canonical_json_sha256(protocol),
            "manifest_path": public_path(manifest_path),
            "manifest_raw_sha256": sha256_file(manifest_path),
            "manifest_semantic_sha256": canonical_json_sha256(manifest),
            "analysis_summary_schema_version": summary["schema_version"],
            "official_run_count": len(expected_keys),
            "full_item_level_run_count": len(full_run_keys),
            "task_metrics": statistics.MC_PRIMARY_METRICS,
            "expected_task_sample_counts": statistics.EXPECTED_TASK_SAMPLE_COUNTS,
            "note": "Reanalysis only; this artifact cannot rerun model inference.",
        },
        "protocol": {
            "permutation_seeds": list(statistics.PERMUTATION_SEEDS),
            "full_task_seeds": list(statistics.FULL_TASK_SEEDS),
            "primary_k": statistics.PRIMARY_K,
            "bootstrap_seed": statistics.BOOTSTRAP_SEED,
            "bootstrap_iterations": statistics.BOOTSTRAP_ITERATIONS,
        },
        "identities": identities,
        "runs": runs,
    }
    bundle["metadata"]["identities_sha256"] = canonical_json_sha256(identities)
    bundle["metadata"]["runs_sha256"] = canonical_json_sha256(runs)
    return bundle


def validate_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported evidence schema: {bundle.get('schema_version')!r}.")
    identities = bundle.get("identities")
    runs = bundle.get("runs")
    if not isinstance(identities, dict) or not isinstance(runs, dict):
        raise ValueError("Evidence bundle must contain identities and runs.")
    if not isinstance(identities.get("wikitext"), list):
        raise ValueError("Evidence bundle is missing WikiText identities.")
    metadata = bundle.get("metadata", {})
    if (
        "identities_sha256" in metadata
        and metadata["identities_sha256"] != canonical_json_sha256(identities)
    ):
        raise ValueError("Evidence identity hash mismatch.")
    if "runs_sha256" in metadata and metadata["runs_sha256"] != canonical_json_sha256(runs):
        raise ValueError("Evidence run-data hash mismatch.")

    seen_by_task: dict[str, set[str]] = {}
    for task, task_identities in identities.items():
        seen: set[str] = set()
        for identity in task_identities:
            if not isinstance(identity, list) or len(identity) != 2:
                raise ValueError(f"Invalid identity in {task}: {identity!r}.")
            key = identity_key(identity)
            if key in seen:
                raise ValueError(f"Duplicate identity in {task}: {identity!r}.")
            seen.add(key)
        seen_by_task[task] = seen

    for run_key, run in runs.items():
        wikitext = run.get("wikitext")
        if not isinstance(wikitext, list) or len(wikitext) != len(identities["wikitext"]):
            raise ValueError(f"WikiText length mismatch for {run_key}.")
        reconstruct_wikitext_ppl(wikitext)
        for task, values in run.get("tasks", {}).items():
            if task not in FULL_SAMPLE_TASKS:
                raise ValueError(f"Unexpected task in evidence bundle: {task}.")
            if len(values) != len(seen_by_task[task]):
                raise ValueError(f"{task} length mismatch for {run_key}.")
            aggregate_from_binary(values)


def compare_hash_bindings(
    *,
    bundle: dict[str, Any],
    summary: dict[str, Any],
    official_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    metadata = bundle["metadata"]
    expected_semantic = {
        "summary_semantic_sha256": canonical_json_sha256(summary),
        "official_records_semantic_sha256": canonical_json_sha256(official_rows),
        "protocol_semantic_sha256": canonical_json_sha256(protocol),
        "manifest_semantic_sha256": canonical_json_sha256(manifest),
    }
    mismatches = {
        key: [metadata.get(key), value]
        for key, value in expected_semantic.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Evidence source semantic hash mismatch: {mismatches}")


def validate_evidence_against_manifest(
    *,
    bundle: dict[str, Any],
    summary: dict[str, Any],
    official_records: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    expected_keys = [config["run_key"] for config in manifest["configs"]]
    runs = bundle["runs"]
    if set(runs) != set(expected_keys):
        raise ValueError("Evidence run keys do not match manifest.")
    if bundle["metadata"].get("official_run_count") != len(expected_keys):
        raise ValueError("Evidence official run count does not match manifest.")
    inventory = summary["analysis_provenance"]["input_runs"]
    if set(inventory) != set(expected_keys):
        raise ValueError("Published summary inventory does not match manifest.")

    for config in manifest["configs"]:
        run_key = config["run_key"]
        run = runs[run_key]
        record = official_records.get(run_key)
        if record is None:
            raise ValueError(f"Missing official record for {run_key}.")
        published = inventory[run_key]
        if run.get("run_id") != published["run_id"] or record.get("run_id") != run.get("run_id"):
            raise ValueError(f"run_id mismatch for {run_key}.")
        if run.get("source_record_sha256") != published["record_sha256"]:
            raise ValueError(f"source record hash mismatch for {run_key}.")
        if run.get("sample_log_sha256") != published["sample_log_sha256"]:
            raise ValueError(f"sample log hash mismatch for {run_key}.")
        if run.get("sample_counts") != published["sample_counts"]:
            raise ValueError(f"sample count metadata mismatch for {run_key}.")
        expected_counts = {
            task: statistics.EXPECTED_TASK_SAMPLE_COUNTS[task]
            for task in config["tasks"]
        }
        if run.get("sample_counts") != expected_counts:
            raise ValueError(f"manifest task/count mismatch for {run_key}.")
        if len(run["wikitext"]) != statistics.EXPECTED_TASK_SAMPLE_COUNTS["wikitext"]:
            raise ValueError(f"WikiText row count mismatch for {run_key}.")
        expected_ppl = statistics.wikitext_word_perplexity(record)
        reconstructed_ppl = reconstruct_wikitext_ppl(run["wikitext"])
        if not math.isclose(reconstructed_ppl, expected_ppl, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"WikiText PPL mismatch for {run_key}.")

        expected_task_set = set(config["tasks"]) - {"wikitext"}
        actual_task_set = set(run.get("tasks", {}))
        if actual_task_set != expected_task_set:
            raise ValueError(f"item task set mismatch for {run_key}.")
        for task in sorted(expected_task_set):
            values = run["tasks"][task]
            if len(values) != statistics.EXPECTED_TASK_SAMPLE_COUNTS[task]:
                raise ValueError(f"{task} row count mismatch for {run_key}.")
            observed = aggregate_from_binary(values)
            expected = statistics.aggregate_metric(record, task, metric_key(task))
            if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{task} aggregate mismatch for {run_key}.")


def write_minimal_sample_logs(
    bundle: dict[str, Any],
    records: dict[str, dict[str, Any]],
    directory: Path,
) -> dict[str, dict[str, Any]]:
    identities = bundle["identities"]
    replay_records: dict[str, dict[str, Any]] = {}
    for run_key, run in bundle["runs"].items():
        if run_key not in records:
            raise ValueError(f"Evidence run {run_key} is absent from official records.")
        record = json.loads(json.dumps(records[run_key]))
        if canonical_json_sha256(record) != run["published_record_sha256"]:
            raise ValueError(f"Official record hash mismatch during replay for {run_key}.")
        sample_path = directory / f"{run_key.replace(':', '__')}.jsonl"
        with sample_path.open("w", encoding="utf-8", newline="\n") as handle:
            for identity, pair in zip(
                identities["wikitext"], run["wikitext"], strict=True
            ):
                handle.write(
                    json.dumps(
                        {
                            "run_id": record["run_id"],
                            "task": "wikitext",
                            "sample": {
                                "doc_id": identity[0],
                                "doc_hash": identity[1],
                                "word_perplexity": pair,
                            },
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            for task, values in run.get("tasks", {}).items():
                key = metric_key(task)
                for identity, value in zip(identities[task], values, strict=True):
                    handle.write(
                        json.dumps(
                            {
                                "run_id": record["run_id"],
                                "task": task,
                                "sample": {
                                    "doc_id": identity[0],
                                    "doc_hash": identity[1],
                                    key: value,
                                },
                            },
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
        record["sample_log_path"] = str(sample_path)
        replay_records[run_key] = record
    return replay_records


def compare_values(actual: Any, expected: Any, path: str = "$") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{path}: expected dict, got {type(actual).__name__}.")
        if set(actual) != set(expected):
            raise AssertionError(
                f"{path}: key mismatch missing={sorted(set(expected) - set(actual))} "
                f"extra={sorted(set(actual) - set(expected))}."
            )
        for key in sorted(expected):
            compare_values(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list length/type mismatch.")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            compare_values(actual_item, expected_item, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        actual_value = float(actual)
        expected_value = float(expected)
        if math.isnan(expected_value):
            if not math.isnan(actual_value):
                raise AssertionError(f"{path}: expected NaN, got {actual!r}.")
            return
        if math.isinf(expected_value):
            if actual_value != expected_value:
                raise AssertionError(f"{path}: expected {expected_value}, got {actual_value}.")
            return
        if not math.isclose(
            actual_value, expected_value, rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL
        ):
            raise AssertionError(f"{path}: {actual_value!r} != {expected_value!r}.")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}.")


def replay_summary(
    *,
    bundle: dict[str, Any],
    official_records: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pruning-evidence-") as temp:
        sample_dir = Path(temp) / "samples"
        sample_dir.mkdir()
        records = write_minimal_sample_logs(bundle, official_records, sample_dir)
        report = {
            "primary_permutation": statistics.primary_permutation_analysis(records, protocol),
            "paired_document_bootstrap": statistics.paired_document_bootstrap(records),
            "exploratory_k2": statistics.exploratory_k2_analysis(records),
            "dose_response": statistics.dose_response_analysis(records, protocol),
            "full_task_aggregate_summary": statistics.full_task_aggregate_summary(records),
            "mcnemar_appendix": statistics.full_task_mcnemar(records),
            "full_task_bootstrap_cis": statistics.full_task_bootstrap_cis(records),
        }
        report.update(recompute_bi_sections(protocol))
        return report


def verify_bundle(
    *,
    bundle_path: Path,
    summary_path: Path,
    official_records_path: Path,
    protocol_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    bundle_sha256 = verify_bundle_sidecar(bundle_path)
    bundle = read_bundle(bundle_path)
    validate_bundle(bundle)
    for key in ("identities_sha256", "runs_sha256"):
        if key not in bundle.get("metadata", {}):
            raise ValueError(f"Evidence bundle metadata missing {key}.")
    summary = read_json(summary_path)
    official_rows = read_jsonl(official_records_path)
    official_records = load_official_records(official_records_path)
    protocol = read_json(protocol_path)
    manifest = read_json(manifest_path)
    compare_hash_bindings(
        bundle=bundle,
        summary=summary,
        official_rows=official_rows,
        protocol=protocol,
        manifest=manifest,
    )
    validate_evidence_against_manifest(
        bundle=bundle,
        summary=summary,
        official_records=official_records,
        manifest=manifest,
    )
    replayed = replay_summary(
        bundle=bundle, official_records=official_records, protocol=protocol
    )
    for section in COMPARE_SECTIONS:
        compare_values(replayed[section], summary[section], f"summary.{section}")
    return {
        "status": "succeeded",
        "bundle": public_path(bundle_path),
        "sha256": bundle_sha256,
        "official_run_count": len(bundle["runs"]),
        "full_item_level_run_count": bundle["metadata"]["full_item_level_run_count"],
        "replayed_sections": list(COMPARE_SECTIONS),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export or verify portable sufficient statistics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for subcommand in ("export", "verify"):
        sub = subparsers.add_parser(subcommand)
        sub.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
        sub.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
        sub.add_argument("--official-records", type=Path, default=DEFAULT_OFFICIAL_RECORDS)
        sub.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        sub.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        if subcommand == "export":
            sub.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "bundle_path": resolve_repo_path(args.bundle),
        "summary_path": resolve_repo_path(args.summary),
        "official_records_path": resolve_repo_path(args.official_records),
        "protocol_path": resolve_repo_path(args.protocol),
        "manifest_path": resolve_repo_path(args.manifest),
    }
    if args.command == "export":
        bundle = build_bundle(
            summary_path=paths["summary_path"],
            official_records_path=paths["official_records_path"],
            protocol_path=paths["protocol_path"],
            manifest_path=paths["manifest_path"],
        )
        write_bundle(paths["bundle_path"], bundle, force=args.force)
        first_hash = sha256_file(paths["bundle_path"])
        with tempfile.TemporaryDirectory(prefix="pruning-evidence-determinism-") as temp:
            check_path = Path(temp) / paths["bundle_path"].name
            write_bundle(check_path, bundle)
            second_hash = sha256_file(check_path)
        if first_hash != second_hash:
            paths["bundle_path"].unlink(missing_ok=True)
            raise RuntimeError("Bundle export is not deterministic.")
        result = {
            "status": "succeeded",
            "bundle": public_path(paths["bundle_path"]),
            "sha256": first_hash,
            "run_count": len(bundle["runs"]),
            "full_item_level_run_count": bundle["metadata"]["full_item_level_run_count"],
        }
    else:
        result = verify_bundle(**paths)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
