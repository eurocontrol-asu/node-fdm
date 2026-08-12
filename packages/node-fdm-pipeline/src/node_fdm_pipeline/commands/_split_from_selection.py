"""Carry the stratified split from the selection CSV into the Delta table.

``fdm split`` assigns ``meta_split`` from a hash of ``raw_icao24``. That belongs
to an older ordering, where the split was drawn after the trajectories had been
processed. In the v2 pipeline the split is drawn **before anything is fetched**,
by ``stratify.py``, and ``ONBOARDING.md`` is explicit about why the key matters:

    The split is keyed on **MSN, never icao24** — nine CRJ airframes carry two
    Mode-S addresses, and an address-keyed split puts one aircraft on both sides
    while a leak check still reads zero.

Hashing ``raw_icao24`` is exactly the failure that warns against. It is not
hypothetical: across the 22 selections **103 MSNs carry more than one icao24**
(47 on CRJ-700 alone). A hash-assigned split would put one airframe in train and
test at once, and a leak check comparing icao24 would see nothing wrong.

It would also discard the stratification itself — ``stratify.py`` balances train
across year, hour, duration band and region, and draws val/test uniformly so the
test set answers "how does this do in service" rather than "how does this do on
the distribution I built".

So this module does not re-derive anything. It reads the split already decided
and joins it on, keeping ``meta_split`` as the column name because the downstream
(``train``, ``predict``, ``visualize``, ``stats``) filters on it.

**Joining on icao24 alone is sound, and the reason is worth stating.** MSN is the
correct split key, but the Delta has no MSN column — it carries the Mode-S
address. What makes the reduction safe is that ``stratify.py`` already propagated
one split per airframe to all of its addresses: across all 22 cohorts, **0 of
3,478 icao24 values map to more than one split** (asserted below, not assumed).
Joining on icao24 therefore reproduces the MSN-keyed split exactly. If that ever
stopped holding, the assertion fails rather than silently splitting an airframe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

log = structlog.get_logger()

__all__ = ["SplitOutcome", "split_from_selection"]


@dataclass(frozen=True)
class SplitOutcome:
    """What the join produced."""

    rows: int
    matched: int
    unmatched: int
    per_split: dict[str, int]

    @property
    def coverage(self) -> float:
        """Fraction of rows that received a split."""
        return self.matched / self.rows if self.rows else 0.0


def _read_selection_splits(selection: Path):
    """Read ``{icao24: split}`` from a selection CSV, refusing an ambiguous map.

    Raises:
        SystemExit: If an icao24 appears under more than one split. That means
            the selection was not MSN-keyed — joining on icao24 would silently
            assign one airframe to two splits, the exact leak ``stratify.py``
            exists to prevent.
    """
    import polars as pl

    frame = pl.read_csv(selection, infer_schema=False)
    for column in ("icao24", "split"):
        if column not in frame.columns:
            raise SystemExit(f"{selection}: no '{column}' column; is this a stratified selection?")

    ambiguous = (
        frame.group_by("icao24")
        .agg(pl.col("split").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if ambiguous.height:
        offenders = ambiguous["icao24"].to_list()[:5]
        raise SystemExit(
            f"{selection}: {ambiguous.height} icao24 appear in several splits "
            f"(e.g. {offenders}). The selection is not MSN-keyed; refusing to join."
        )

    return frame.select("icao24", "split").unique()


def split_from_selection(
    *,
    config: Path,
    selection: Path,
    dry_run: bool = False,
) -> SplitOutcome | None:
    """Write ``meta_split`` into the Delta table from a stratified selection.

    Args:
        config: Path to the cohort's YAML config.
        selection: Path to the cohort's ``selection_*.csv``.
        dry_run: Report the coverage the join would achieve, write nothing.

    Returns:
        The outcome, or ``None`` on a dry run.

    Raises:
        SystemExit: If the selection is not MSN-keyed, or if some rows would be
            left without a split — a null ``meta_split`` silently drops flights
            from every downstream filter, so it is refused rather than written.
    """
    import polars as pl
    from node_fdm_data.delta import read_delta_table, write_columns

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    delta_table = cfg.paths.resolve("delta_table")
    splits = _read_selection_splits(selection)

    log.info(
        "split_from_selection_start",
        table=str(delta_table),
        selection=str(selection),
        aircraft=splits.height,
    )

    df = read_delta_table(delta_table)
    if "meta_split" in df.columns:
        log.info("split_from_selection_drop_existing")
        df = df.drop("meta_split")

    joined = df.join(
        splits.rename({"icao24": "raw_icao24", "split": "meta_split"}),
        on="raw_icao24",
        how="left",
    )

    matched = int(joined["meta_split"].is_not_null().sum())
    unmatched = joined.height - matched
    per_split = {
        row["meta_split"]: row["len"]
        for row in joined.group_by("meta_split").len().iter_rows(named=True)
        if row["meta_split"] is not None
    }
    outcome = SplitOutcome(
        rows=joined.height, matched=matched, unmatched=unmatched, per_split=per_split
    )

    if unmatched:
        missing = (
            joined.filter(pl.col("meta_split").is_null())["raw_icao24"].unique().to_list()[:5]
        )
        raise SystemExit(
            f"{unmatched:,} of {joined.height:,} rows have no split "
            f"(icao24 not in the selection, e.g. {missing}). "
            "Decode and selection disagree on which aircraft this cohort covers."
        )

    if dry_run:
        log.info(
            "split_from_selection_dry_run",
            rows=outcome.rows,
            coverage=f"{outcome.coverage:.1%}",
            per_split=outcome.per_split,
        )
        return None

    write_columns(joined, delta_table)
    log.info(
        "split_from_selection_done",
        rows=outcome.rows,
        coverage=f"{outcome.coverage:.1%}",
        per_split=outcome.per_split,
    )
    return outcome
