# DPP measurement records: semantic validation

**Status:** project adapter for GitHub issue #54
**Upstream:** Digital Presence Protocol 0.3.0 Working Draft, revision `38eb016986731bed3eaa7aecfc6ba14f3fbc8ccd`, vendored byte-for-byte in [`schemas/vendor/dpp-0.3.0/`](../../schemas/vendor/dpp-0.3.0/) with SHA-256 hashes in `SOURCE.json`.

The pinned measurement schemas check the shape of a record, not its meaning. This repository adds the missing meaning in [`scripts/dpp_measurement.py`](../../scripts/dpp_measurement.py) without editing the upstream files. Passing either check does not establish DPP conformance.

## What the pinned schema accepts or rejects

Reproduced by `tests/test_dpp_measurement.py` against the vendored bytes (the probes need the `jsonschema` package and are skipped where it is absent, as in CI):

| Probe | Pinned schema | Why it matters |
|---|---|---|
| Metric with 3 successes out of 1 opportunity | Accepted | A rate cannot exceed its denominator. |
| Empty `entity_id`, `claim_id`, `representation_id`, `surface_id` | Accepted | The record cannot be traced to anything. |
| `finding_level: causal` with no experiment | Accepted | Causal claims need an assessed experiment; the schema has no field to reference one. |
| An explicit `time_window` field | Rejected (`additionalProperties: false`) | Observation windows, criteria and missingness have no place in the record. |
| An all-zero placeholder `sha256` | Accepted | The evidence hash cannot be resolved. The upstream example uses one. |
| `claim_id` on a representation-level observation (for example availability) | Required | Such observations measure a surface, not a claim, so they must anchor to a claim they do not test. |

## The project adapter

An observation stays a valid upstream record. Project facts travel beside it:

```json
{
  "observation": { "…": "upstream measurement-observation record, unchanged" },
  "binding": {
    "time_window": { "start": "2026-11-03T13:06:19Z", "end": "2026-11-03T13:06:40Z" },
    "criteria": "HTTP 200 with a self-referencing canonical for each sitemap URL",
    "missingness": "none",
    "experiment_ref": null
  }
}
```

`scripts/dpp_measurement.py ENVELOPE.json [--experiments DIR] [--artifacts DIR]` reports errors and GRAY items and exits 1 on any error. It checks that:

- the four identifiers are non-empty, and entity, claim and representation IDs resolve against [`docs/DPP_IMPLEMENTATION.md`](../DPP_IMPLEMENTATION.md);
- the window has ISO 8601 start and end in order, and the observation is not dated before it;
- criteria and missingness are stated (`none` is a valid statement);
- a metric's successes do not exceed its opportunities, and a metric with no opportunities is only GRAY or NOT_APPLICABLE;
- a causal finding, or an experiment-mode observation, references an experiment record that resolves, and a causal finding's experiment is `completed`, declares `causal_claim: causal` and has results;
- evidence hashes are not placeholders, and local artifacts match their recorded hash.

[`examples/dpp/availability-2026-11-03.json`](../../examples/dpp/availability-2026-11-03.json) is a real record: the 2026-11-03 census of all ten sitemap URLs, with evidence pointing at the committed sitemap by its hash.

## Ownership

| Change | Owner |
|---|---|
| Binding envelope, semantic checks, identifier registry, example | This repository (done) |
| `minLength: 1` on the four identifiers | Proposed upstream fix |
| `numerator ≤ denominator` (not expressible in JSON Schema alone; needs a rule in the protocol text and validators) | Proposed upstream fix |
| Optional `time_window`, `criteria`, `missingness` and `experiment_ref` fields, or a documented extension point | Proposed upstream fix |
| `experiment_ref` required when `finding_level` is `causal` (JSON Schema `if`/`then`) | Proposed upstream fix |
| Reject placeholder hashes; replace the all-zero hash in the upstream example | Proposed upstream fix |
| Make `claim_id` optional for representation-level observations | Proposed upstream fix |

No upstream file was edited. When the protocol publishes fixes, re-vendor the new revision, update `SOURCE.json` and `.agents/project/formal-resources.json`, and drop any adapter rule the schema then enforces.

## Still GRAY

- Surface IDs (Google, Bing, public HTTP and so on) have no declared register, so they are checked only for being non-empty.
- Remote evidence (`https://…`) is not fetched, so its hash is unchecked and reported as GRAY.
- Metric dictionary units and denominators are not cross-checked against `DPP_IMPLEMENTATION.md`'s metric table.
- No observation has been recorded under DPP-MPF conformance; this adapter is a validation aid, not a conformance claim.
