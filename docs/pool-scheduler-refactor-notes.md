# Pool Scheduler Refactor Notes

## Why This Refactor Exists

The current pool API and scheduler expose more request-level routing policy than the service should likely keep long-term.

Two concepts now feel different in quality:

- `interactive` still feels like a real pool-level responsibility
- `normal`, `background`, and `routing.slot_affinity` feel more like legacy operational policy

`interactive` maps to a stable user-facing need:

- latency-sensitive work
- fairness between active sessions
- protection against batch-like work consuming all slots

By contrast, `normal` versus `background` mainly encode scheduler policy that came out of an older hardware and capacity situation.

`routing.slot_affinity` is also not a good long-term fit for the public request contract. In practice it was closer to `slot_required` than to real affinity, and it leaked pool topology into request payloads.


## Current State

Today the request contract exposes:

- `priority`: `interactive`, `normal`, `background`
- `routing.fairness_key`
- `routing.slot_affinity`

Today the scheduler also has extra policy attached to those choices:

- interactive requests get fairness handling
- background requests are throttled more aggressively
- one background request may run at a time
- one special-case slot-affinity path still exists

This works, but the API is carrying scheduler history rather than just stable request intent.


## Desired Direction

The request contract should express only durable request intent.

The pool configuration should own capacity policy.

That implies the following direction:

- keep `interactive`
- collapse `normal` and `background` into one non-interactive mode
- remove `routing.slot_affinity`
- move interactive capacity protection into pool startup config

The core distinction then becomes:

- interactive request
- non-interactive request

The request should not choose a slot.
The pool should decide how many slots must remain available for interactive work.


## Target Model

### Request Semantics

Requests should eventually expose only:

- `priority: "interactive"` for latency-sensitive work
- otherwise the default non-interactive behavior
- optional `routing.fairness_key` for interactive fairness grouping

The request should not contain:

- slot numbers
- hardware-lane assumptions
- pool-topology routing hints


### Pool Semantics

Pool startup config should declare the interactive reservation policy.

The important question is not "which slot does this request want?"

The important question is:

- how many slots exist
- how many must remain available for interactive work
- that reserved interactive slots remain unavailable to non-interactive work

A possible configuration shape:

```json
{
  "scheduler": {
    "runner_slots": 6,
    "interactive_reserved_slots": 1
  }
}
```

This is conceptually cleaner than `slot_affinity` because it keeps capacity policy inside the pool instead of pushing it into every request.


## Why `interactive` Still Belongs

`interactive` is not just a scheduler implementation detail.

It reflects a meaningful product-level distinction:

- low-latency user-facing requests
- fairness between sessions
- stronger protection from bulk work

That is a stable reason for the pool to keep a dedicated public `priority: "interactive"` mode.


## Why `normal` And `background` Likely Do Not

`normal` and `background` currently encode:

- different queue buckets
- different concurrency treatment

But those differences do not appear to represent stable external intent in the same way that `interactive` does.

They look more like local scheduler policy.

If the pool still needs different treatment for some non-interactive work later, that should first be justified as a stable product concept, not preserved only because the historical scheduler happened to use it.


## Why `slot_affinity` Should Go

`routing.slot_affinity` has several problems:

- it leaks slot topology into the request contract
- it was effectively used as a required-slot hint, not real affinity
- it ties public request semantics to a specific hardware layout
- it becomes harder to reason about once slot counts or reservation policy change

This makes it a poor long-term API concept even if it was useful as an operational workaround.


## Likely End State

### Contract

- keep `priority: "interactive"`
- default everything else to one non-interactive mode
- keep `routing.fairness_key`
- remove `routing.slot_affinity`


### Scheduler

- keep interactive fairness
- replace background-specific throttling with general non-interactive scheduling policy
- add explicit interactive slot reservation in config
- let the scheduler, not the request, decide slot placement


### Here Is How It Should Work

The pool should own these decisions internally:

- reserve one or more slots for interactive work
- reserved slots are interactive-only
- reserved interactive slots are never lent to non-interactive work

Those are pool policy decisions, not request-routing primitives.


## Suggested Refactor Order

1. Add pool startup config for interactive reservation.
2. Collapse `background` into `normal` so both non-interactive request types go through the `normal` path.
3. Once `background` no longer has its own path, remove the old `background` path completely: scheduler branches, queue handling, config keys, metadata fields, API examples, and any other dead code or compatibility behavior.
4. Remove `routing.slot_affinity` from the request contract and lifecycle metadata.
5. Update README and API examples only after the code path is actually simplified.
