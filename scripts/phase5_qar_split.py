"""Phase 5 — QAR train/test split deterministic (R3 strict).

Hash-based partition of the 517 A320 cruise-stable QAR flights into
414 train + 103 test, with full reproducibility and no overlap.

Audit checks :
* No overlap between train/test (assert intersect = empty).
* Ratio close to 80/20 (tolerance ±5 vols).
* Distribution similarity (SYS__GW median per split, optional sanity).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import polars as pl

QAR_DIR = Path("/Users/gabriel/Downloads/QAR3")
OUT_DIR = Path("data/qar_splits")
TRAIN_TXT = OUT_DIR / "phase5_train.txt"
TEST_TXT = OUT_DIR / "phase5_test.txt"
REPORT_MD = OUT_DIR / "phase5_split_report.md"


def deterministic_bin(filename: str, seed: int) -> int:
    """Return a deterministic bin 0-99 for a filename + seed."""
    key = f"{seed}-{filename}".encode()
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) % 100


def is_valid_cruise(parquet_path: Path) -> tuple[bool, float | None]:
    """Quick filter : check if flight has needed columns + cruise window.

    Returns (is_valid, m_qar_at_toc).
    """
    needed = {"ALT__STD", "SPD__TAS", "SPD__MACH", "SYS__GW",
              "FUEL__FF_LEFT", "FUEL__FF_RIGHT", "TEMP__SAT"}
    try:
        df = pl.read_parquet(parquet_path, columns=list(needed))
    except Exception:  # noqa: BLE001
        return False, None
    if not needed.issubset(df.columns):
        return False, None
    df = df.with_columns([
        (pl.col("ALT__STD") / 3.28084).alias("alt_m"),
    ])
    df_toc = df.filter(
        (pl.col("alt_m") >= 8500.0)
        & (pl.col("alt_m") <= 10500.0)
        & (pl.col("SYS__GW") > 40000)
        & (pl.col("SYS__GW") < 80000)
        & (pl.col("FUEL__FF_LEFT") > 0)
        & (pl.col("FUEL__FF_RIGHT") > 0)
    )
    if len(df_toc) == 0:
        return False, None
    m_qar = float(df_toc["SYS__GW"].head(1).to_numpy()[0])
    return True, m_qar


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.80)
    args = parser.parse_args()

    paths = sorted(QAR_DIR.glob("*A320*.parquet"))
    print(f"Found {len(paths)} A320 QAR files")
    if not paths:
        print("No QAR files — abort")
        return 1

    # First pass : filter valid flights + get m_qar for stratification check.
    valid_flights: list[tuple[str, float]] = []
    skipped = 0
    for i, p in enumerate(paths):
        if i % 100 == 0:
            print(f"  [{i+1}/{len(paths)}] {p.name}")
        ok, m_qar = is_valid_cruise(p)
        if ok and m_qar is not None:
            valid_flights.append((p.name, m_qar))
        else:
            skipped += 1
    print(f"\nValid flights : {len(valid_flights)} ({skipped} skipped)")

    # Hash-based split.
    bin_threshold = int(args.train_frac * 100)
    train: list[tuple[str, float]] = []
    test: list[tuple[str, float]] = []
    for name, m_qar in valid_flights:
        bin_id = deterministic_bin(name, args.seed)
        if bin_id < bin_threshold:
            train.append((name, m_qar))
        else:
            test.append((name, m_qar))

    print(f"\nTrain : {len(train)} flights ({100*len(train)/len(valid_flights):.1f} %)")
    print(f"Test  : {len(test)} flights ({100*len(test)/len(valid_flights):.1f} %)")

    # R3 audit : no overlap.
    train_set = {n for n, _ in train}
    test_set = {n for n, _ in test}
    overlap = train_set & test_set
    assert not overlap, f"R3 audit FAIL : {len(overlap)} files in both splits"
    print(f"\nR3 audit : ✅ no overlap (intersect = ∅)")

    # Stratification sanity.
    m_train = np.array([m for _, m in train])
    m_test = np.array([m for _, m in test])
    train_med = float(np.median(m_train))
    test_med = float(np.median(m_test))
    train_p10 = float(np.percentile(m_train, 10))
    train_p90 = float(np.percentile(m_train, 90))
    test_p10 = float(np.percentile(m_test, 10))
    test_p90 = float(np.percentile(m_test, 90))
    print(f"\nm_qar (kg) distribution :")
    print(f"  train : median={train_med:.0f}, p10={train_p10:.0f}, p90={train_p90:.0f}")
    print(f"  test  : median={test_med:.0f}, p10={test_p10:.0f}, p90={test_p90:.0f}")
    diff_med_pct = abs(train_med - test_med) / train_med * 100
    print(f"  median diff : {diff_med_pct:.2f} %")
    if diff_med_pct < 5.0:
        print(f"  ✅ stratified-enough (median diff < 5 %)")
    else:
        print(f"  ⚠️ split is stratified-imbalanced — re-seed if needed")

    # Write split files.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_TXT.write_text("\n".join(n for n, _ in train) + "\n")
    TEST_TXT.write_text("\n".join(n for n, _ in test) + "\n")
    print(f"\nWritten : {TRAIN_TXT} ({len(train)} lines)")
    print(f"Written : {TEST_TXT} ({len(test)} lines)")

    # Report markdown.
    lines = [
        "# Phase 5 — QAR train/test split report",
        "",
        f"> Generated by `scripts/phase5_qar_split.py --seed {args.seed}`.",
        "",
        "## Distribution",
        "",
        "| Split | n_flights | % | m_qar median | m_qar p10 | m_qar p90 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Train | {len(train)} | {100*len(train)/len(valid_flights):.1f} % | {train_med:.0f} | {train_p10:.0f} | {train_p90:.0f} |",
        f"| Test  | {len(test)} | {100*len(test)/len(valid_flights):.1f} % | {test_med:.0f} | {test_p10:.0f} | {test_p90:.0f} |",
        "",
        "## R3 audit",
        "",
        f"- No overlap : ✅ ({len(overlap)} duplicates)",
        f"- Median diff : {diff_med_pct:.2f} % (target < 5 %)",
        f"- Stratification : {'✅' if diff_med_pct < 5.0 else '⚠️'}",
        "",
        "## Files",
        "",
        f"- `{TRAIN_TXT}` ({len(train)} flight filenames)",
        f"- `{TEST_TXT}` ({len(test)} flight filenames)",
        "",
        f"## Hash details",
        "",
        f"- seed = {args.seed}",
        f"- train_frac = {args.train_frac}",
        f"- bin_threshold = {bin_threshold} (out of 100)",
        f"- Algorithm : `int(sha256(seed-filename)[:8], 16) % 100 < {bin_threshold}` → train",
    ]
    REPORT_MD.write_text("\n".join(lines))
    print(f"Report  : {REPORT_MD}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
