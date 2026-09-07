# Fleet lease coordination

Fleet campaigns use one shared filesystem lease to serialize the OpenSky boundary across
processes and machines. The lease record is still created exclusively and is never shared
between owners; waiting wraps acquisition without weakening that invariant.

## Waiting policy

`FleetRunConfig` carries two coordination settings:

- `lease_wait_budget_s` is the maximum elapsed wait before the campaign gives up;
- `lease_poll_interval_s` is the upper bound between acquisition attempts.

While a live owner holds the record, the campaign waits for the smaller of the poll
interval, the remaining budget, and the holder's remaining lease lifetime. It retries
after release or expiry. Once the budget is exhausted, `LeaseUnavailable` is surfaced
with the current holder's name. A zero budget preserves immediate refusal for callers
that do not opt into waiting.

The lease heartbeat continues at one third of the TTL. An expired record may still be
replaced, and a stale owner cannot renew or release its successor's record.

## Campaign preflight

The central campaign preflight accepts exactly `plan`, `run`, `resume`, `status`, and
`validate`. A live `run` or `resume` requires both a shared `lease_path` and a positive
`min_free_gib` in `FleetRunConfig`; missing values are rejected before remote I/O. The
field stays optional on the model because non-live modes do not need the live guard.

A new `run` may start without recorded state, but refuses an existing campaign identity
when its selection, resolved configuration, or profile digest differs. `resume` instead
requires that identity and rejects any digest divergence. These checks reuse the durable
resume digest and shared-lease validation used by fleet acquisition.

## CLI configuration

A bounded campaign invocation can add these options to the normal six campaign inputs:

```bash
fdm download-fleet \
  --fleet-dir fleet \
  --selection run/selection.json \
  --resolved-config run/resolved-config.json \
  --profile run/profile.json \
  --lease-path /shared/node-fdm/download-fleet.lease \
  --lease-ttl-s 120 \
  --lease-wait-budget-s 300 \
  --lease-poll-interval-s 1 \
  --disk-min-gib 8.5
```

If the budget expires, the command exits non-zero and writes an error such as
`LeaseUnavailable: Lease is held by campaign-a` to standard error.

## Campaign selection

For campaign downloads, `--selection` may contain a JSON array of source rows:

```json
[
  {
    "icao24": "a00001",
    "callsign": "ALPHA1",
    "firstseen": 1577835000,
    "lastseen": 1577838600,
    "msn": "M1",
    "split": "train",
    "cohort": "C1",
    "selection_id": "sel-alpha",
    "utc_days": ["20191231", "20200101"]
  },
  {
    "icao24": "a00001",
    "callsign": "ALPHA1",
    "firstseen": 1577835000,
    "lastseen": 1577838600,
    "msn": "M1",
    "split": "train",
    "cohort": "C2",
    "selection_id": "sel-alpha",
    "utc_days": ["20191231", "20200101"]
  }
]
```

Rows with the same flight identity are compiled into one acquisition shared by their
cohorts. Explicit `utc_days` values are authoritative; when omitted, they are derived
from `firstseen` and `lastseen`. The resulting plan ignores aircraft and dates
present only in the per-cohort selection CSV files, while the exact JSON text remains the
resume-digest input.

Each selected day requests `history`, `extended`, and `flightlist` in
that order. One staging artifact is published per (UTC day, kind), with all owning cohorts
recorded in its consumer ledger.

## Acquisition journal

Pass both `--acquisition-journal` and `--acquisition-receipt-dir` to persist
coordination evidence. Each journal line points to an atomic JSON receipt. Wait events use
state `waiting` and include both `owner` and `holder`. Recorded-source campaigns also
emit `boundary_entered` after acquisition and `lease_released` after cleanup, allowing
consumers to reconstruct the maximum number of simultaneously open OpenSky sections.

These coordination observations are separate from the durable run lifecycle states, so
existing `acquiring`, `processing`, failure, interruption, and cleanup semantics remain
unchanged.

## Recorded OpenSky source

`--recorded-source` selects an offline JSON source instead of the live provider. The
format contains an optional non-negative delay and one response per acquisition kind:

```json
{
  "delay_s": 0.25,
  "responses": {
    "history": null,
    "extended": null,
    "flightlist": null
  }
}
```

A response is either `null` for a recorded zero-row result or a list of row objects.
History, extended, and flight-list responses are replayed in the same sequential order as
the live boundary, with `cached=False` semantics preserved. This is a deployment feature
for deterministic replay and incident reproduction, not a test-only hook.
