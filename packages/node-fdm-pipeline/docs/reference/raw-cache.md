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

## Campaign identification

When an `identify` configuration has the adjacent
`results/selection_<campaign>.csv` file, the command compiles that selection and passes
the staged rotations to `match_selections`. Only uniquely admitted rotations are written
to the identified output; each row receives the selection's `selection_id`, canonical
`callsign`, interval, `msn`, `split`, `cohorts`, and `utc_days`. Rotations outside
the selected callsign and time interval are not copied merely because their aircraft was
staged for the same day.

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

Before a fleet campaign submits any acquisition, it reconciles every planned `(day, kind)`
against the staged absence artefact and receipt. If a process stopped after publishing an
intact payload but before publishing its receipt, reconciliation derives the expected digest
from the planned aircraft set, validates the parquet, and republishes only the receipt. The
day remains visible and no acquisition boundary is called.

A truncated payload, an unreadable receipt, a missing referenced payload, or a digest
mismatch makes that `(day, kind)` invalid. Its absence artefacts and receipt are removed
before the day is submitted exactly once for acquisition, allowing the two-phase publication
to restart from a clean state. A fully verified day remains untouched.

`read_absence_receipt(root, day, kind)` reloads and validates the JSON receipt from disk.
No process-local state is required. It returns `None` when no receipt has been published.

## Cache semantics

`is_cached(cfg, kind, day, icao24)` remains true when the normal parquet partition exists.
When it does not, the function checks an acquisition absence receipt and returns true only
if that aircraft belongs to the receipt’s covered set. `cache_misses` inherits the same
semantics for a batch; identification receipts are audit evidence for selection matching,
not aircraft cache-hit evidence.
