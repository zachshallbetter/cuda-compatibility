from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.dpp_measurement import (
    VENDOR,
    load_schema,
    registry_from_dpp_record,
    schema_errors,
    semantic_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALID_OBSERVATION = {
    "observation_id": "OBS-TEST-001",
    "entity_id": "cuda-compatibility",
    "claim_id": "CLC-CLAIM-001",
    "representation_id": "site-home",
    "surface_id": "google-search",
    "observed_at": "2026-11-03T13:06:19Z",
    "mode": "observation",
    "retrieval_path": "reported",
    "finding_level": "descriptive",
    "verdict": "GRAY",
    "metric": None,
    "evidence": [{"artifact_ref": "routes.txt", "sha256": "1f49789e450b42af7ccf193988a0205588ccef66bc28a165332c2e782b9b4959", "evidence_grade": "E1"}],
    "limitations": [],
}
VALID_ENVELOPE = {
    "observation": VALID_OBSERVATION,
    "binding": {
        "time_window": {"start": "2026-11-03T00:00:00Z", "end": "2026-11-03T13:00:00Z"},
        "criteria": "HTTP 200 for a canonical sitemap URL",
        "missingness": "none",
        "experiment_ref": None,
    },
}


def envelope(**changes: object) -> dict:
    record = copy.deepcopy(VALID_ENVELOPE)
    for key, value in changes.items():
        target = record["binding"] if key in record["binding"] else record["observation"]
        target[key] = value
    return record


class VendoredSchemaTests(unittest.TestCase):
    def test_vendored_schemas_match_the_pinned_hashes(self) -> None:
        source = json.loads((VENDOR / "SOURCE.json").read_text(encoding="utf-8"))
        self.assertEqual(source["revision"], "38eb016986731bed3eaa7aecfc6ba14f3fbc8ccd")
        for name, entry in source["files"].items():
            self.assertEqual(hashlib.sha256((VENDOR / name).read_bytes()).hexdigest(), entry["sha256"], name)


@unittest.skipIf(schema_errors({}, {}) is None, "jsonschema is not installed; upstream shape probes skipped")
class UpstreamSchemaProbeTests(unittest.TestCase):
    """Reproduce the reported gaps against the exact pinned schema."""

    def setUp(self) -> None:
        self.schema = load_schema("measurement-observation")

    def accepts(self, record: dict) -> bool:
        return schema_errors(record, self.schema) == []

    def test_baseline_record_is_accepted(self) -> None:
        self.assertTrue(self.accepts(VALID_OBSERVATION))

    def test_schema_accepts_more_successes_than_opportunities(self) -> None:
        record = dict(VALID_OBSERVATION, metric={
            "metric_id": "DPP-AV-01", "numerator": 3, "denominator": 1, "unit": "proportion",
            "sampling_frame": "sitemap URLs", "uncertainty_method": "none",
        })
        self.assertTrue(self.accepts(record))

    def test_schema_accepts_empty_identifiers(self) -> None:
        self.assertTrue(self.accepts(dict(VALID_OBSERVATION, entity_id="", claim_id="", representation_id="", surface_id="")))

    def test_schema_accepts_causal_label_without_experiment(self) -> None:
        self.assertTrue(self.accepts(dict(VALID_OBSERVATION, finding_level="causal")))

    def test_schema_rejects_an_explicit_time_window(self) -> None:
        self.assertFalse(self.accepts(dict(VALID_OBSERVATION, time_window={"start": "2026-11-01T00:00:00Z"})))


class SemanticCheckTests(unittest.TestCase):
    def errors(self, record: dict, **kwargs: object) -> list[str]:
        return semantic_report(record, **kwargs).errors

    def test_valid_envelope_passes_and_stays_gray_for_conformance(self) -> None:
        report = semantic_report(VALID_ENVELOPE)
        self.assertEqual(report.errors, [])
        self.assertIn("passing these checks does not establish DPP conformance", report.gray)

    def test_rejects_more_successes_than_opportunities(self) -> None:
        metric = {"metric_id": "DPP-AV-01", "numerator": 3, "denominator": 1, "unit": "proportion",
                  "sampling_frame": "sitemap URLs", "uncertainty_method": "none"}
        self.assertTrue(any("3 successes out of 1" in e for e in self.errors(envelope(metric=metric))))

    def test_rejects_satisfied_verdict_with_no_opportunities(self) -> None:
        metric = {"metric_id": "DPP-AV-01", "numerator": 0, "denominator": 0, "unit": "proportion",
                  "sampling_frame": "none", "uncertainty_method": "none"}
        self.assertTrue(any("no opportunities" in e for e in self.errors(envelope(metric=metric, verdict="SATISFIED"))))

    def test_rejects_empty_and_unresolved_identifiers(self) -> None:
        self.assertTrue(any("entity_id must be" in e for e in self.errors(envelope(entity_id=""))))
        known = registry_from_dpp_record(PROJECT_ROOT / "docs" / "DPP_IMPLEMENTATION.md")
        self.assertTrue(any("does not resolve" in e for e in self.errors(envelope(claim_id="CLC-CLAIM-999"), known_ids=known)))
        self.assertEqual(self.errors(VALID_ENVELOPE, known_ids=known), [])

    def test_requires_a_well_formed_observation_window(self) -> None:
        self.assertTrue(any("time_window needs" in e for e in self.errors(envelope(time_window=None))))
        backwards = {"start": "2026-11-03T12:00:00Z", "end": "2026-11-01T00:00:00Z"}
        self.assertTrue(any("starts after it ends" in e for e in self.errors(envelope(time_window=backwards))))
        self.assertTrue(any("binding.criteria" in e for e in self.errors(envelope(criteria=""))))

    def test_causal_label_needs_a_completed_assessed_experiment(self) -> None:
        causal = envelope(finding_level="causal", experiment_ref="EXP-1")
        self.assertTrue(any("resolvable binding.experiment_ref" in e for e in self.errors(causal)))
        running = {"EXP-1": {"status": "running", "causal_claim": "causal", "results": None}}
        self.assertTrue(any("not a completed, assessed" in e for e in self.errors(causal, experiments=running)))
        done = {"EXP-1": {"status": "completed", "causal_claim": "causal", "results": {"effect": 0.1}}}
        self.assertEqual(self.errors(causal, experiments=done), [])

    def test_rejects_placeholder_and_mismatched_artifact_hashes(self) -> None:
        placeholder = envelope(evidence=[{"artifact_ref": "x", "sha256": "0" * 64, "evidence_grade": "E0"}])
        self.assertTrue(any("placeholder hash" in e for e in self.errors(placeholder)))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "routes.txt"
            artifact.write_text("observed\n", encoding="utf-8")
            good = hashlib.sha256(artifact.read_bytes()).hexdigest()
            matching = envelope(evidence=[{"artifact_ref": "routes.txt", "sha256": good, "evidence_grade": "E1"}])
            self.assertEqual(self.errors(matching, artifact_root=Path(directory)), [])
            self.assertTrue(any("does not match" in e for e in self.errors(VALID_ENVELOPE, artifact_root=Path(directory))))


if __name__ == "__main__":
    unittest.main()
