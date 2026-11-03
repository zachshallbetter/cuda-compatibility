# Execution Pipeline

## Coordinator loop

```mermaid
flowchart LR
    A[Preflight] --> B[Reconcile]
    B --> C[Compute Ready frontier]
    C --> D[Select and validate]
    D --> E[Claim and isolate]
    E --> F[Compile context]
    F --> G[Execute]
    G --> H[Candidate]
    H --> I[Evaluate]
    I --> J{Qualification required?}
    J -->|Yes| K[Qualify]
    J -->|No| L[Apply authorized state]
    K --> L
    L --> M[Record]
    M --> N[Release claim]
    N --> B
```

## Local blocker semantics

A worker may return:

```text
DONE
BLOCKED
FAILED
SUPERSEDED
ABANDONED
WAITING_EXTERNAL
```

The coordinator then asks:

```text
Does this invalidate project-wide authority/safety?
  yes → GLOBAL STOP / human decision
  no  → update graph and recompute Ready frontier
```

Do not turn a local blocker into a conversational project halt.

```mermaid
flowchart TD
    Worker[Worker disposition] --> Check{Does blocker invalidate project authority or safety?}
    Check -->|Yes| Stop[Global stop and human decision]
    Check -->|No| Recompute[Update work graph and recompute Ready frontier]
    Recompute --> Continue[Continue unrelated Ready work]
```

## Retry

Retry only when:

```text
failure class is retryable
∧ new information or changed state exists
∧ declared recovery exists
∧ retry budget remains
```

Never repeatedly send an unchanged blocker to the same worker.

## Goal sequencing

Before mutation, identify:

```text
prerequisites
ordered work
verification
promotion/evidence
cleanup
```

Continue through obvious lawful successors without waiting for a new prompt.

## Findings outside scope

Classify:

```text
in-scope
related-out-of-scope
decision-required
```

Fix only in-scope findings. Preserve the rest as work/evidence.

## Human decision boundary

Escalation packets should contain:

```text
decision needed
why existing rules cannot decide
affected work
available alternatives
evidence
consequence of no decision
smallest authority change required
```

The human should resolve the governing state, not micromanage implementation.

## Model routing

Use the smallest model that can satisfy the work contract.

```text
deterministic tool / validator first
small model for mechanical bounded work
larger model for unresolved architecture/strategy/contradiction
```

Model tier never grants authority.
