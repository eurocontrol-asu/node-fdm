# Raw-cache campaign staging and absence receipts

The raw cache records a successful zero-row acquisition as an explicit artefact. This
distinguishes “the provider returned no rows for this batch” from “this batch has never
been acquired”.

## Campaign staging

A fleet campaign stages raw payloads once under the campaign root rather than copying
them into each cohort's data directory. `cache_root(cfg, kind)` resolves to
`<campaign-root>/raw/<kind>` when the configuration carries the absolute campaign
cache anchor produced by fleet-plan construction. Standalone configurations keep the
existing `<data-dir>/raw/<kind>` layout.

History and extended payloads retain the partition layout
`date=<day>/icao24=<icao24>/data.parquet`. When several cohorts select the same
aircraft-day, acquisition writes that parquet artefact once. The day directory also
contains `consumers.json`, an atomically published ledger mapping every consuming
cohort to `"pending"`. The file is flushed, renamed, and its directory synchronized
before it becomes visible.

## Contract

`absence_digest(day, kind, icao24s)` computes a SHA-256 digest from the day, acquisition
kind, and the sorted, deduplicated aircraft set. Input order therefore does not affect
the digest, while a different covered set produces a different value.

`publish_absence(root, day, kind, icao24s)` publishes two durable files:

- a zero-row parquet artefact under `date=<day>/.absences/<digest>.parquet`;
- a JSON receipt under `.absences/<day>.<kind>.json`.

The receipt contains `day`, `kind`, `digest`, `row_count` (always `0`), and the canonical
`icao24s` tuple. The parquet artefact is written first through the cache’s atomic writer;
the receipt is exposed last through an atomic rename. A crash cannot make a partial
receipt visible as valid coverage.

`read_absence_receipt(root, day, kind)` reloads and validates the JSON receipt from disk.
No process-local state is required. It returns `None` when no receipt has been published.

## Cache semantics

`is_cached(cfg, kind, day, icao24)` remains true when the normal parquet partition exists.
When it does not, the function checks the persisted absence receipt and returns true only
if that aircraft belongs to the receipt’s covered set. `cache_misses` inherits the same
semantics for a batch.
