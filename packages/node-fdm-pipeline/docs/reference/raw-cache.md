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

## Trino failures and partial day receipts

`classify_trino_failure(error)` normalizes provider failures into a
`TrinoFailure(code, kind, raw_message)`. It reads a native Trino
`error_name` when one is available and otherwise falls back to the exception
message. The acquisition action is:

- `retry_later` for `QUERY_QUEUE_FULL`;
- `split` for `EXCEEDED_TIME_LIMIT` and `EXCEEDED_MEMORY_LIMIT`;
- `terminal` for every other code.

A split is processed from its first child to its second child. If a child hits
the configured bisection floor and fails again, the other children still finish.
The resulting `DateOutcome` has `kind="terminal_floor"` and
`failing_batch` identifies only the indivisible failed child. Its
`staged_icao24s` contains only payloads that were actually published, and
`staged_digest` is the SHA-256 digest of those payload bytes concatenated in
that same order. The `date_finished` manifest event exposes these fields as the
durable day receipt.

Each `fetch_attempt` journal event also records the `acquisition_keys` of the
selections served by its batch. `replay_attempts` preserves journal order,
attempt rank, outcome, observed backoff delay, batch, and selection attribution.
Legacy events without attribution replay with an empty key tuple.

## Campaign identification

`identify` accepts `--selection <path>` for an explicit recorded campaign selection in
JSON-row or CSV format. The explicit file takes precedence; when the option is omitted,
the command still discovers an adjacent `results/selection_<campaign>.csv` file. It
compiles the selected source and passes the staged rotations to `match_selections`. Only
uniquely admitted rotations are written to the identified output; each row receives the
selection's `selection_id`, `icao24`, canonical `callsign`, interval, `msn`, `split`,
`cohorts`, and `utc_days`. Callsigns are matched despite case or surrounding whitespace,
and the selected canonical value is written. Rotations outside the selected callsign and
time interval are not copied merely because their aircraft was staged for the same day.

The command returns the structured `MatchResult` for campaign runs. After the identified
output has been written successfully, every `absent` rejection is published through
`publish_absence` for the expected campaign day and the `history` kind. These
identification receipts record selection IDs (for example `sel-zulu`) as their covered
identifiers, making the absence replayable without conflating it with an admitted row.
Standalone identification without an adjacent selection keeps the legacy behavior and
returns `None`.

## Contract

`absence_digest(day, kind, icao24s)` computes a SHA-256 digest from the day, acquisition
kind, and the sorted, deduplicated covered identifier set. Acquisition receipts pass
aircraft identifiers; campaign-identification receipts pass selection IDs. Input order
therefore does not affect the digest, while a different covered set produces a different
value.

`publish_absence(root, day, kind, icao24s)` publishes two durable files:

- a zero-row parquet artefact under `date=<day>/.absences/<digest>.parquet`;
- a JSON receipt under `.absences/<day>.<kind>.json`.

The receipt contains `day`, `kind`, `digest`, `row_count` (always `0`), and the canonical
covered identifiers in the legacy-named `icao24s` tuple. The parquet artefact is written
first through the cache’s atomic writer; the receipt is exposed last through an atomic
rename. Both boundaries flush their file before the rename and synchronize the containing
directory afterwards. Temporary payload names are unique, so concurrent writers cannot
share an incomplete file. A crash cannot make a partial receipt visible as valid coverage.

## Start-up reconciliation

Before a fleet campaign submits any acquisition, it loads the receipt for every planned
`(day, kind)` and reconciles it with the staged absence artefact. A normal aircraft parquet
does not certify a completed partition by itself: if that parquet exists but its receipt is
missing, the day is submitted exactly once for acquisition and the receipt is republished.
The narrower crash-recovery rule for a standalone, intact zero-row absence artefact remains:
reconciliation can derive its expected digest and republish only its receipt.

A truncated payload, an unreadable receipt, a missing referenced payload, or a digest
mismatch makes that `(day, kind)` invalid. Its absence artefacts and receipt are removed
before reacquisition. When the receipt, digest, covered identifiers, and artefact agree, the
campaign issues no boundary call and appends a `date_reused` manifest event containing the
`date`, `kind`, and `receipt_digest`.

For campaign-selection receipts, covered selection IDs are also restored as
`DateOutcome.rejections`. A resumed `sel-zulu` absence therefore returns a
`MatchRejection(selection_id="sel-zulu", kind="absent")` instead of querying the
provider again.

`read_absence_receipt(root, day, kind)` reloads and schema-validates the JSON receipt from
disk. No process-local state is required. It returns `None` when no receipt has been
published; cache admission separately verifies its day, kind, digest, covered identifiers,
and referenced artefact.

## Cache semantics

`is_cached(cfg, kind, day, identifier)` is receipt-gated: a normal parquet partition alone
is never a verified cache hit. The function returns true only when a durable receipt has the
requested day and kind, its digest matches its canonical covered set, the identifier belongs
to that set, and the referenced absence artefact exists. `cache_misses` applies the same
rule to a batch.

Fleet plans translate aircraft IDs to selection IDs before this check, so identification
receipts can suppress and restitute a previously proven absence. The legacy standalone
download and decode paths continue to inspect their parquet staging directly; they do not
silently acquire the fleet campaign's receipt-based resume semantics.
