#!/usr/bin/env python3
"""Validate prediction pipeline output (Neural ODE predictions).

Checks prediction coverage, column names, physical plausibility,
NaN absence, and produces altitude/TAS overlay plots (GT vs Neural ODE).

Run:
    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/validate_05_prediction.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config() -> tuple[Path, Path, Path, Path, list[str]]:
    """Load pipeline config and return resolved paths + typecodes."""
    sys.path.insert(0, str(PROJECT_ROOT / "packages" / "node-fdm-pipeline" / "src"))
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(PROJECT_ROOT / "config.yaml")
    predicted_dir = cfg.paths.resolve("predicted_dir")
    process_dir = cfg.paths.resolve("process_dir")
    figure_dir = cfg.paths.data_dir / "figures"
    split_csv = process_dir / "dataset_split.csv"
    return predicted_dir, process_dir, figure_dir, split_csv, cfg.typecodes


def _status(ok: bool, msg: str) -> bool:
    """Print a status line and return the check result."""
    icon = "✅" if ok else "❌"
    print(f"  {icon} {msg}")
    return ok


# ---------------------------------------------------------------------------
# T1 — Prediction coverage (AC1)
# ---------------------------------------------------------------------------


def validate_coverage(pred_dir: Path, split_csv: Path) -> tuple[bool, set[str], set[str]]:
    """Check that prediction files cover the test set.

    Returns:
        Tuple of (check passed, predicted flight IDs, test flight IDs).
    """
    split = pl.read_csv(split_csv)
    test_flights = split.filter(pl.col("split") == "test")
    test_ids = set(test_flights["flight_id"].to_list())

    pred_files = {f.stem for f in pred_dir.glob("*.parquet")}
    coverage = len(pred_files & test_ids) / len(test_ids) if test_ids else 0.0

    ok = coverage >= 0.9
    _status(ok, f"Coverage: {len(pred_files)}/{len(test_ids)} flights ({coverage:.1%})")

    missing = test_ids - pred_files
    if missing:
        _status(len(missing) <= 5, f"Missing: {len(missing)} flights")

    return ok, pred_files, test_ids


# ---------------------------------------------------------------------------
# T2 — Column names (AC2)
# ---------------------------------------------------------------------------

EXPECTED_COLS = {"pred_distance_m", "pred_altitude_m", "pred_gamma_rad", "pred_tas_ms"}


def validate_columns(pred_dir: Path, pred_ids: set[str]) -> bool:
    """Check prediction parquet column names on a sample file."""
    sample_id = next(iter(pred_ids))
    pred = pl.read_parquet(pred_dir / f"{sample_id}.parquet")
    actual = set(pred.columns)

    ok = EXPECTED_COLS <= actual
    _status(ok, f"Columns: {sorted(actual)}")
    if not ok:
        _status(False, f"Missing columns: {EXPECTED_COLS - actual}")

    return ok


# ---------------------------------------------------------------------------
# T3 — Physical plausibility (AC3)
# ---------------------------------------------------------------------------


def validate_physics(pred_dir: Path, pred_ids: set[str], n_sample: int = 10) -> bool:
    """Assert physical bounds on a random sample of predicted flights."""
    sample = random.sample(sorted(pred_ids), min(n_sample, len(pred_ids)))
    ok = True

    for fid in sample:
        p = pl.read_parquet(pred_dir / f"{fid}.parquet")

        # Drop NaN before bounds check (NaN is caught separately in T4)
        p_clean = p.drop_nulls().filter(~pl.any_horizontal(pl.all().is_nan()))
        if len(p_clean) == 0:
            ok = _status(False, f"{fid}: all rows are NaN — cannot check physics") and ok
            continue

        alt = p_clean["pred_altitude_m"]
        alt_ok = (alt.min() > -100) and (alt.max() < 20_000)  # type: ignore[operator]
        if not alt_ok:
            ok = _status(False, f"{fid}: altitude out of [-100, 20000] m") and ok

        tas = p_clean["pred_tas_ms"]
        tas_ok = (tas.min() >= 0) and (tas.max() < 400)  # type: ignore[operator]
        if not tas_ok:
            ok = _status(False, f"{fid}: TAS out of [0, 400] m/s") and ok

        gamma = p_clean["pred_gamma_rad"]
        gamma_ok = (gamma.min() > -0.5) and (gamma.max() < 0.5)  # type: ignore[operator]
        if not gamma_ok:
            ok = _status(False, f"{fid}: gamma out of [-0.5, 0.5] rad") and ok

    _status(ok, f"Physics plausible on {len(sample)} sampled flights")
    return ok


# ---------------------------------------------------------------------------
# T4 — No NaN (AC4)
# ---------------------------------------------------------------------------


def validate_no_nan(pred_dir: Path) -> bool:
    """Scan all prediction files for NaN values."""
    files = sorted(pred_dir.glob("*.parquet"))
    nan_flights: list[str] = []

    for f in files:
        p = pl.read_parquet(f)
        null_total = sum(p[col].null_count() for col in p.columns)
        # Also check for NaN in float columns (polars treats NaN != null)
        nan_total = sum(p[col].is_nan().sum() for col in p.columns if p[col].dtype.is_float())
        if null_total > 0 or nan_total > 0:
            nan_flights.append(f.stem)

    ok = len(nan_flights) == 0
    if ok:
        _status(True, f"Zero NaN/null across {len(files)} files")
    else:
        _status(False, f"NaN/null found in {len(nan_flights)}/{len(files)} files")
        for fid in nan_flights[:5]:
            print(f"    → {fid}")
        if len(nan_flights) > 5:
            print(f"    → ... and {len(nan_flights) - 5} more")

    return ok


# ---------------------------------------------------------------------------
# Plots — Altitude & TAS overlays (AC5, AC6)
# ---------------------------------------------------------------------------


def plot_overlays(pred_dir: Path, process_dir: Path, figure_dir: Path, pred_ids: set[str]) -> bool:
    """Generate GT vs Neural ODE overlay plots for a sample flight."""
    import altair as alt

    sample_id = sorted(pred_ids)[0]
    gt_path = process_dir / "flights" / f"{sample_id}.parquet"
    pred_path = pred_dir / f"{sample_id}.parquet"

    if not gt_path.exists():
        return _status(False, f"GT file not found: {gt_path}")

    gt = pl.read_parquet(gt_path)
    pred = pl.read_parquet(pred_path)

    n = min(len(gt), len(pred))

    # GT uses imperial units — convert to SI for comparison
    gt_alt_m = (gt["altitude_ft"][:n] * 0.3048).to_list()
    gt_tas_ms = (gt["tas_kt"][:n] * 0.514444).to_list()

    # --- Altitude overlay ---
    combined_alt = pl.DataFrame(
        {
            "t": list(range(n)) * 2,
            "altitude_m": gt_alt_m + pred["pred_altitude_m"][:n].to_list(),
            "source": ["Ground Truth"] * n + ["Neural ODE"] * n,
        }
    )

    chart_alt = (
        alt.Chart(combined_alt.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Timestep (x4s)"),
            y=alt.Y("altitude_m:Q", title="Altitude (m)"),
            color="source:N",
            strokeDash="source:N",
        )
        .properties(
            title=f"Altitude — GT vs Neural ODE — {sample_id}",
            width=700,
            height=300,
        )
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    alt_path = figure_dir / "05_altitude_overlay.html"
    chart_alt.save(alt_path)
    _status(True, f"Altitude overlay saved to {alt_path}")

    # --- TAS overlay ---
    combined_tas = pl.DataFrame(
        {
            "t": list(range(n)) * 2,
            "tas_ms": gt_tas_ms + pred["pred_tas_ms"][:n].to_list(),
            "source": ["Ground Truth"] * n + ["Neural ODE"] * n,
        }
    )

    chart_tas = (
        alt.Chart(combined_tas.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Timestep (x4s)"),
            y=alt.Y("tas_ms:Q", title="TAS (m/s)"),
            color="source:N",
            strokeDash="source:N",
        )
        .properties(
            title=f"TAS — GT vs Neural ODE — {sample_id}",
            width=700,
            height=300,
        )
    )

    tas_path = figure_dir / "05_tas_overlay.html"
    chart_tas.save(tas_path)
    _status(True, f"TAS overlay saved to {tas_path}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all prediction validation checks."""
    print("=" * 60)
    print("  Prediction Validation (05)")
    print("=" * 60)

    predicted_dir, process_dir, figure_dir, split_csv, typecodes = _load_config()

    all_ok = True

    for acft in typecodes:
        pred_dir = predicted_dir / acft
        print(f"\n{'─' * 60}")
        print(f"  Typecode: {acft} — {pred_dir}")
        print(f"{'─' * 60}")

        if not pred_dir.exists():
            _status(False, f"Prediction directory not found: {pred_dir}")
            all_ok = False
            continue

        print("\n--- T1: Coverage ---")
        t1_ok, pred_ids, _test_ids = validate_coverage(pred_dir, split_csv)

        if not pred_ids:
            _status(False, "No predictions found — skipping remaining checks")
            all_ok = False
            continue

        print("\n--- T2: Columns ---")
        t2_ok = validate_columns(pred_dir, pred_ids)

        print("\n--- T3: Physical plausibility ---")
        t3_ok = validate_physics(pred_dir, pred_ids)

        print("\n--- T4: No NaN ---")
        t4_ok = validate_no_nan(pred_dir)

        print("\n--- Plots: GT vs Neural ODE overlays ---")
        plot_overlays(pred_dir, process_dir, figure_dir, pred_ids)

        all_ok = all_ok and t1_ok and t2_ok and t3_ok and t4_ok

    # Summary
    print("\n" + "=" * 60)
    icon = "✅" if all_ok else "❌"
    print(f"  {icon} Overall: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
