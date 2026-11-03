"""Project-side semantic checks for DPP 0.3.0 measurement observations.

The pinned upstream schemas (schemas/vendor/dpp-0.3.0) check shape only. This module adds the
meaning they cannot express, without editing them. An observation stays a valid upstream record;
the project's extra facts travel beside it in a binding envelope:

    {
      "observation": { ...upstream measurement-observation record... },
      "binding": {
        "time_window": {"start": "2026-11-01T00:00:00Z", "end": "2026-11-03T00:00:00Z"},
        "criteria": "what counts as a success for this metric",
        "missingness": "what was not observable, or 'none'",
        "experiment_ref": "EXP-..." or null
      }
    }

Usage: python3 scripts/dpp_measurement.py ENVELOPE.json [--experiments DIR] [--artifacts DIR]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "schemas" / "vendor" / "dpp-0.3.0"
ID_FIELDS = ("entity_id", "claim_id", "representation_id", "surface_id")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    gray: list[str] = field(default_factory=list)


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((VENDOR / f"{name}.schema.json").read_text(encoding="utf-8"))


def schema_errors(record: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str] | None:
    """Upstream shape errors, or None when the jsonschema package is not installed."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return None
    return [error.message for error in Draft202012Validator(schema).iter_errors(record)]


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def registry_from_dpp_record(path: Path) -> dict[str, set[str]]:
    """Entity, claim and surface IDs declared in the project's DPP implementation record."""
    text = path.read_text(encoding="utf-8")

    def table_ids(heading: str) -> set[str]:
        section = text.split(heading, 1)[1].split("\n## ", 1)[0] if heading in text else ""
        return set(re.findall(r"^\| `([^`]+)`", section, re.M))

    surfaces = {part.strip() for ids in table_ids("## Representation and surface register") for part in ids.split(",")}
    return {
        "entity_id": table_ids("## Entity and identity"),
        "claim_id": set(re.findall(r"`(CLC-CLAIM-\d+)`", text)),
        "representation_id": surfaces,
    }


def semantic_report(
    envelope: Mapping[str, Any],
    *,
    experiments: Mapping[str, Mapping[str, Any]] | None = None,
    artifact_root: Path | None = None,
    known_ids: Mapping[str, set[str]] | None = None,
) -> Report:
    report = Report()
    observation = envelope.get("observation")
    binding = envelope.get("binding")
    if not isinstance(observation, Mapping) or not isinstance(binding, Mapping):
        report.errors.append("envelope must contain an 'observation' object and a 'binding' object")
        return report

    for name in ID_FIELDS:
        value = observation.get(name)
        if not isinstance(value, str) or not value.strip():
            report.errors.append(f"{name} must be a non-empty identifier")
        elif known_ids and name in known_ids and value not in known_ids[name]:
            report.errors.append(f"{name} {value!r} does not resolve to a declared identifier")

    observed_at = _timestamp(observation.get("observed_at"))
    window = binding.get("time_window")
    start = _timestamp(window.get("start")) if isinstance(window, Mapping) else None
    end = _timestamp(window.get("end")) if isinstance(window, Mapping) else None
    if start is None or end is None:
        report.errors.append("binding.time_window needs ISO 8601 start and end")
    elif start > end:
        report.errors.append("binding.time_window starts after it ends")
    elif observed_at is not None and observed_at < start:
        report.errors.append("observed_at precedes the observation window")
    for name in ("criteria", "missingness"):
        if not isinstance(binding.get(name), str) or not binding[name].strip():
            report.errors.append(f"binding.{name} must state its value, or 'none'")

    metric = observation.get("metric")
    if isinstance(metric, Mapping):
        numerator, denominator = metric.get("numerator"), metric.get("denominator")
        if isinstance(numerator, int) and isinstance(denominator, int):
            if numerator > denominator:
                report.errors.append(f"metric counts {numerator} successes out of {denominator} opportunities")
            if denominator == 0 and observation.get("verdict") not in {"GRAY", "NOT_APPLICABLE"}:
                report.errors.append("a metric with no opportunities can only be GRAY or NOT_APPLICABLE")

    experiment_ref = binding.get("experiment_ref")
    needs_experiment = observation.get("finding_level") == "causal" or observation.get("mode") == "experiment"
    if needs_experiment:
        experiment = (experiments or {}).get(experiment_ref) if isinstance(experiment_ref, str) else None
        if experiment is None:
            report.errors.append("causal findings and experiment-mode observations need a resolvable binding.experiment_ref")
        elif observation.get("finding_level") == "causal" and not (
            experiment.get("status") == "completed"
            and experiment.get("causal_claim") == "causal"
            and experiment.get("results") not in (None, {})
        ):
            report.errors.append(f"experiment {experiment_ref} is not a completed, assessed causal analysis")

    for item in observation.get("evidence") or []:
        digest, ref = str(item.get("sha256", "")), str(item.get("artifact_ref", ""))
        if len(set(digest)) <= 1:
            report.errors.append(f"evidence {ref!r} carries a placeholder hash")
            continue
        local = (artifact_root / ref) if artifact_root and not re.match(r"^[a-z]+://", ref) else None
        if local is None or not local.is_file():
            report.gray.append(f"evidence {ref!r} was not resolved to a local artifact; its hash is unchecked")
        elif hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            report.errors.append(f"evidence {ref!r} does not match its recorded sha256")
    report.gray.append("passing these checks does not establish DPP conformance")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("envelope", type=Path)
    parser.add_argument("--experiments", type=Path, help="directory of upstream experiment records (*.json)")
    parser.add_argument("--artifacts", type=Path, help="root against which local artifact_ref paths resolve")
    args = parser.parse_args(argv)
    envelope = json.loads(args.envelope.read_text(encoding="utf-8"))
    experiments = {}
    if args.experiments:
        for path in sorted(args.experiments.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            experiments[record.get("experiment_id", path.stem)] = record
    shape = schema_errors(envelope.get("observation", {}), load_schema("measurement-observation"))
    report = semantic_report(
        envelope,
        experiments=experiments,
        artifact_root=args.artifacts,
        known_ids=registry_from_dpp_record(ROOT / "docs" / "DPP_IMPLEMENTATION.md"),
    )
    if shape is None:
        report.gray.append("upstream schema shape not checked: the jsonschema package is not installed")
    else:
        report.errors[:0] = [f"upstream schema: {message}" for message in shape]
    for line in report.errors:
        print(f"ERROR {line}")
    for line in report.gray:
        print(f"GRAY  {line}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
