# Raw-cache absence receipts

The raw cache records a successful zero-row acquisition as an explicit artefact. This
distinguishes “the provider returned no rows for this batch” from “this batch has never
been acquired”.

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
