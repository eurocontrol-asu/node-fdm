"""Train / val / test splitting by ICAO aircraft type.

Groups flight files by ICAO code (extracted from the filename) and
deterministically assigns each group to a split based on the given
ratios.  This ensures all flights of the same aircraft type end up
in the same split, preventing data leakage.

Example::

    result = split_by_icao(Path("output/A320"), ratios=(0.7, 0.15, 0.15))
"""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl

__all__ = [
    "split_by_icao",
]

_MIN_GROUPS_FOR_THREE_WAY = 3
_MIN_GROUPS_FOR_TWO_WAY = 2


def split_by_icao(
    data_dir: Path | str,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> pl.DataFrame:
    """Split flight files into train / val / test by ICAO group.

    File names are expected to contain the ICAO type code as the 4th
    underscore-separated token (e.g. ``flight_001_002_A320_seg.parquet``).

    Args:
        data_dir: Directory containing per-flight files.
        ratios: ``(train, val, test)`` proportions — must sum to ~1.0.
        seed: Random seed for deterministic shuffling.

    Returns:
        ``pl.DataFrame`` with columns ``filepath``, ``icao``, ``split``.
    """
    data_dir = Path(data_dir)

    files = sorted(f.name for f in data_dir.iterdir() if f.is_file() and "_" in f.name)

    if not files:
        return pl.DataFrame(
            {"filepath": [], "icao": [], "split": []},
            schema={"filepath": pl.Utf8, "icao": pl.Utf8, "split": pl.Utf8},
        )

    icaos = [f.split("_")[3] for f in files]
    df = pl.DataFrame(
        {
            "filepath": [str(data_dir / f) for f in files],
            "icao": icaos,
        }
    )

    # Count flights per ICAO and shuffle
    icao_counts: dict[str, int] = {}
    for ic in icaos:
        icao_counts[ic] = icao_counts.get(ic, 0) + 1

    rng = random.Random(seed)  # noqa: S311
    icao_list = list(icao_counts.keys())
    rng.shuffle(icao_list)

    total_flights = len(files)
    train_target = int(total_flights * ratios[0])

    # Assign ICAO groups to splits — ensure at least 1 group per split
    # when there are ≥3 distinct ICAOs.
    train_icaos: set[str] = set()
    val_icaos: set[str] = set()
    test_icaos: set[str] = set()

    n_icaos = len(icao_list)

    if n_icaos >= _MIN_GROUPS_FOR_THREE_WAY:
        # Reserve last two groups for val and test
        assignable = icao_list[: n_icaos - _MIN_GROUPS_FOR_TWO_WAY]
        reserved = icao_list[n_icaos - _MIN_GROUPS_FOR_TWO_WAY :]
    elif n_icaos == _MIN_GROUPS_FOR_TWO_WAY:
        assignable = icao_list[:1]
        reserved = icao_list[1:]
    else:
        # Single ICAO — everything goes to train
        assignable = icao_list
        reserved = []

    # Fill train from assignable groups
    running = 0
    for ic in assignable:
        if running < train_target:
            train_icaos.add(ic)
            running += icao_counts[ic]
        else:
            break

    # Remaining assignable groups go to early pool for val/test
    remaining = [ic for ic in assignable if ic not in train_icaos] + reserved
    remaining_total = sum(icao_counts[ic] for ic in remaining)
    val_target = int(remaining_total * ratios[1] / (ratios[1] + ratios[2])) if remaining else 0

    running = 0
    for ic in remaining:
        if running < val_target:
            val_icaos.add(ic)
            running += icao_counts[ic]
        else:
            test_icaos.add(ic)

    def _assign(icao: str) -> str:
        if icao in train_icaos:
            return "train"
        if icao in val_icaos:
            return "val"
        return "test"

    splits = [_assign(ic) for ic in df["icao"].to_list()]
    return df.with_columns(pl.Series("split", splits))
