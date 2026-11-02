# Recovery

Recovery reconstructs safe operating state from canonical project records rather than hidden conversational memory.

## Recovery triggers

Examples:

```text
stale context lock
stale or orphaned claim
worker termination
branch behind material canonical change
unexpected dirty state
coordination collision
provider state changed beneath a claim
```

## Procedure

```mermaid
flowchart TD
    Trigger[Recovery trigger] --> Stop[Stop unsafe mutation]
    Stop --> Preserve[Preserve WIP, candidate, and evidence]
    Preserve --> Identify[Identify repository, branch, revision, and dirty state]
    Identify --> Verify[Verify context and source identity]
    Verify --> Refresh[Reread live work, claims, and provider state]
    Refresh --> Classify[Classify stale state or ownership conflict]
    Classify --> Context{Context regeneration required?}
    Context -->|Yes| Regenerate[Regenerate context]
    Context -->|No| Claims[Release, reacquire, or transfer claims lawfully]
    Regenerate --> Claims
    Claims --> Checks[Rerun relevant checks]
    Checks --> Resume[Resume from known state]
```

Never erase another worker's unknown state as a shortcut to recovery.

Do not force-reset or delete work merely because it is inconvenient.

## Recovery evidence

Record:

```text
trigger
stale/inconsistent state detected
what was preserved
canonical state reconstructed
ownership changes
context changes
checks rerun
supported resumption point
```
