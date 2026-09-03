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

## CLI configuration

A bounded campaign invocation can add these options to the normal six campaign inputs:

```bash
fdm download-fleet \
  --fleet-dir fleet \
  --selection run/selection.digest \
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
