import gzip
import json
import math

import pytest

from experiments import evidence_bundle


def test_bundle_writer_is_deterministic(tmp_path):
    bundle = {
        "schema_version": 1,
        "artifact_type": "confirmatory_sufficient_statistics",
        "metadata": {},
        "identities": {"wikitext": [[0, "a"]]},
        "runs": {
            "base:baseline:k0:seednone": {
                "wikitext": [[-2.0, 2.0]],
                "published_record_sha256": "record",
                "sample_log_sha256": "sample",
            }
        },
    }
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"

    evidence_bundle.write_bundle(first, bundle)
    evidence_bundle.write_bundle(second, bundle)

    assert evidence_bundle.sha256_file(first) == evidence_bundle.sha256_file(second)
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        assert json.load(handle)["schema_version"] == 1


def test_bundle_writer_refuses_silent_overwrite(tmp_path):
    path = tmp_path / "bundle.json.gz"
    bundle = {"schema_version": 1, "identities": {"wikitext": []}, "runs": {}}
    evidence_bundle.write_bundle(path, bundle)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evidence_bundle.write_bundle(path, bundle)


def test_validate_bundle_rejects_duplicate_identities():
    bundle = {
        "schema_version": 1,
        "identities": {"wikitext": [[0, "same"], [0, "same"]]},
        "runs": {},
    }

    with pytest.raises(ValueError, match="Duplicate identity"):
        evidence_bundle.validate_bundle(bundle)


def test_validate_bundle_rejects_identity_hash_tampering():
    bundle = {
        "schema_version": 1,
        "metadata": {"identities_sha256": "wrong"},
        "identities": {"wikitext": [[0, "same"]]},
        "runs": {},
    }

    with pytest.raises(ValueError, match="Evidence identity hash mismatch"):
        evidence_bundle.validate_bundle(bundle)


def test_validate_bundle_rejects_metric_length_mismatch():
    bundle = {
        "schema_version": 1,
        "identities": {
            "wikitext": [[0, "w0"]],
            "piqa": [[0, "p0"], [1, "p1"]],
        },
        "runs": {
            "run": {
                "wikitext": [[-1.0, 1.0]],
                "tasks": {"piqa": [1.0]},
            }
        },
    }

    with pytest.raises(ValueError, match="piqa length mismatch"):
        evidence_bundle.validate_bundle(bundle)


def test_validate_bundle_rejects_non_binary_task_metric():
    bundle = {
        "schema_version": 1,
        "identities": {"wikitext": [[0, "w0"]], "piqa": [[0, "p0"]]},
        "runs": {
            "run": {
                "wikitext": [[-1.0, 1.0]],
                "tasks": {"piqa": [0.5]},
            }
        },
    }

    with pytest.raises(ValueError, match="Binary task evidence"):
        evidence_bundle.validate_bundle(bundle)


def test_write_minimal_logs_preserves_samples_for_statistics(tmp_path):
    record = {
        "run_id": "rid",
        "sample_log_path": "unused",
        "provenance": {"run_key": "base:baseline:k0:seednone"},
    }
    record_hash = evidence_bundle.canonical_json_sha256(record)
    bundle = {
        "schema_version": 1,
        "identities": {"wikitext": [[0, "w0"]], "piqa": [[0, "p0"]]},
        "runs": {
            "base:baseline:k0:seednone": {
                "published_record_sha256": record_hash,
                "sample_log_sha256": "unused",
                "wikitext": [[-2.0, 2.0]],
                "tasks": {"piqa": [1.0]},
            }
        },
    }

    records = evidence_bundle.write_minimal_sample_logs(
        bundle, {"base:baseline:k0:seednone": record}, tmp_path
    )

    wikitext = evidence_bundle.statistics.load_task_samples(
        records["base:baseline:k0:seednone"], "wikitext"
    )
    piqa = evidence_bundle.statistics.load_task_samples(
        records["base:baseline:k0:seednone"], "piqa"
    )
    assert next(iter(wikitext.values()))["word_perplexity"] == [-2.0, 2.0]
    assert next(iter(piqa.values()))["acc_norm"] == 1.0


def test_compare_values_reports_tampering():
    with pytest.raises(AssertionError, match="root.value"):
        evidence_bundle.compare_values({"value": 1.0}, {"value": 2.0}, "root")


def test_source_binding_uses_semantic_hashes_for_tracked_json(tmp_path):
    summary = {"schema_version": 2, "value": 1}
    official_rows = [{"run_id": "a"}]
    protocol = {"protocol": True}
    manifest = {"configs": []}
    summary_path = tmp_path / "summary.json"
    official_path = tmp_path / "official_runs.jsonl"
    protocol_path = tmp_path / "protocol.json"
    manifest_path = tmp_path / "manifest.json"
    summary_path.write_text('{\r\n  "schema_version": 2,\r\n  "value": 1\r\n}\r\n')
    official_path.write_text('{"run_id":"a"}\r\n')
    protocol_path.write_text('{"protocol":true}\r\n')
    manifest_path.write_text('{"configs":[]}\r\n')
    bundle = {
        "metadata": {
            "summary_raw_sha256": "not-portable",
            "summary_semantic_sha256": evidence_bundle.canonical_json_sha256(summary),
            "official_records_raw_sha256": "not-portable",
            "official_records_semantic_sha256": evidence_bundle.canonical_json_sha256(
                official_rows
            ),
            "protocol_raw_sha256": "not-portable",
            "protocol_semantic_sha256": evidence_bundle.canonical_json_sha256(protocol),
            "manifest_raw_sha256": "not-portable",
            "manifest_semantic_sha256": evidence_bundle.canonical_json_sha256(manifest),
        }
    }

    evidence_bundle.compare_hash_bindings(
        bundle=bundle,
        summary=summary,
        official_rows=official_rows,
        protocol=protocol,
        manifest=manifest,
    )


def test_sidecar_rejects_payload_tampering(tmp_path):
    path = tmp_path / "bundle.json.gz"
    bundle = {"schema_version": 1, "identities": {"wikitext": []}, "runs": {}}
    evidence_bundle.write_bundle(path, bundle)
    path.write_bytes(path.read_bytes() + b"x")

    with pytest.raises(ValueError, match="Evidence bundle SHA mismatch"):
        evidence_bundle.verify_bundle_sidecar(path)


def _record(run_key, ppl, piqa=None):
    results = {"wikitext": {"word_perplexity,none": ppl}}
    if piqa is not None:
        results["piqa"] = {"acc_norm,none": piqa}
    return {
        "run_id": f"run-{run_key}",
        "provenance": {"run_key": run_key},
        "results": {"results": results},
    }


def _inventory(record, sample_counts):
    return {
        "run_id": record["run_id"],
        "record_sha256": "source-record",
        "sample_log_sha256": "sample-log",
        "sample_counts": sample_counts,
    }


def test_manifest_validation_rejects_tampered_k8_wikitext():
    run_key = "base:random:k8:seed3"
    record = _record(run_key, 1.0)
    sample_counts = {"wikitext": 62}
    bundle = {
        "metadata": {"official_run_count": 1},
        "identities": {"wikitext": [[index, f"w{index}"] for index in range(62)]},
        "runs": {
            run_key: {
                "run_id": record["run_id"],
                "source_record_sha256": "source-record",
                "sample_log_sha256": "sample-log",
                "sample_counts": sample_counts,
                "wikitext": [[-1.0, 1.0] for _ in range(62)],
            }
        },
    }
    manifest = {"configs": [{"run_key": run_key, "tasks": ["wikitext"]}]}
    summary = {"analysis_provenance": {"input_runs": {run_key: _inventory(record, sample_counts)}}}
    bundle["runs"][run_key]["wikitext"][0][0] = -2.0

    with pytest.raises(ValueError, match="WikiText PPL mismatch"):
        evidence_bundle.validate_evidence_against_manifest(
            bundle=bundle,
            summary=summary,
            official_records={run_key: record},
            manifest=manifest,
        )


def test_manifest_validation_rejects_tampered_k2_binary_correctness():
    run_key = "base:random:k2:seed0"
    record = _record(run_key, math.e, piqa=1.0)
    sample_counts = {"piqa": 1838, "wikitext": 62}
    piqa_values = [1.0] * 1838
    bundle = {
        "metadata": {"official_run_count": 1},
        "identities": {
            "wikitext": [[index, f"w{index}"] for index in range(62)],
            "piqa": [[index, f"p{index}"] for index in range(1838)],
        },
        "runs": {
            run_key: {
                "run_id": record["run_id"],
                "source_record_sha256": "source-record",
                "sample_log_sha256": "sample-log",
                "sample_counts": sample_counts,
                "wikitext": [[-1.0, 1.0] for _ in range(62)],
                "tasks": {"piqa": piqa_values},
            }
        },
    }
    manifest = {"configs": [{"run_key": run_key, "tasks": ["piqa", "wikitext"]}]}
    summary = {"analysis_provenance": {"input_runs": {run_key: _inventory(record, sample_counts)}}}
    bundle["runs"][run_key]["tasks"]["piqa"][0] = 0.0

    with pytest.raises(ValueError, match="piqa aggregate mismatch"):
        evidence_bundle.validate_evidence_against_manifest(
            bundle=bundle,
            summary=summary,
            official_records={run_key: record},
            manifest=manifest,
        )


def test_manifest_validation_rejects_wrong_run_count():
    run_key = "base:random:k8:seed3"
    record = _record(run_key, 1.0)
    sample_counts = {"wikitext": 62}
    bundle = {
        "metadata": {"official_run_count": 2},
        "identities": {"wikitext": [[index, f"w{index}"] for index in range(62)]},
        "runs": {
            run_key: {
                "run_id": record["run_id"],
                "source_record_sha256": "source-record",
                "sample_log_sha256": "sample-log",
                "sample_counts": sample_counts,
                "wikitext": [[-1.0, 1.0] for _ in range(62)],
            }
        },
    }
    manifest = {"configs": [{"run_key": run_key, "tasks": ["wikitext"]}]}
    summary = {"analysis_provenance": {"input_runs": {run_key: _inventory(record, sample_counts)}}}

    with pytest.raises(ValueError, match="official run count"):
        evidence_bundle.validate_evidence_against_manifest(
            bundle=bundle,
            summary=summary,
            official_records={run_key: record},
            manifest=manifest,
        )


def test_bi_recompute_accepts_lf_checkout_and_rejects_changed_scores(tmp_path, monkeypatch):
    source_root = evidence_bundle.REPO_ROOT
    protocol = evidence_bundle.read_json(source_root / "experiments/permutation_protocol.json")
    temp_root = tmp_path / "repo"
    (temp_root / "experiments/bi").mkdir(parents=True)
    for name in ("base.json", "instruct.json", "math.json"):
        source = source_root / "experiments/bi" / name
        target = temp_root / "experiments/bi" / name
        target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    monkeypatch.setattr(evidence_bundle, "REPO_ROOT", temp_root)

    recomputed = evidence_bundle.recompute_bi_sections(protocol)

    assert recomputed["bi_overlap"] == protocol["bi_overlap"]
    assert recomputed["canonical_legacy_spearman"] == pytest.approx(
        {
            model_key: protocol["models"][model_key]["canonical_legacy_spearman"]
            for model_key in evidence_bundle.statistics.MODEL_KEYS
        }
    )

    base_path = temp_root / "experiments/bi/base.json"
    changed = base_path.read_text(encoding="utf-8").replace(
        "0.844884523190558", "0.844884523190559"
    )
    base_path.write_text(changed, encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="Frozen text hash mismatch"):
        evidence_bundle.recompute_bi_sections(protocol)
