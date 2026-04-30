"""Round 2B — Phase classification / boundary proximity diagnostic.

Hypothesis under test (H_2B)
----------------------------
Les samples descent_shallow `tas_known=0` sont concentres aux frontieres
de phase (debut/fin de la descente) ou en zones isolees (long ecart
temporel depuis le dernier `tas_known=1`), ou le contexte dynamique est
mal defini -- la classification descent_shallow est partiellement
erronee pour ces samples, et le bias +19.73 m/s reflete ce mismatch
de contexte.

Falsifier
---------
- Statistique primaire: fraction des samples `known=0` descent_shallow
  qui sont SOIT (a) a moins de 5% de la frontiere temporelle de la phase
  de descente (au debut ou a la fin), SOIT (b) "isoles" -- pas de sample
  `known=1` dans une fenetre de +/-60s (15 samples a 4s).
- Threshold: H_2B survives si fraction > 50% sur l'agregat.
  Dies si < 20%. Indetermine entre.
- Null world: si H_2B etait fausse, les samples `known=0` descent_shallow
  seraient distribues uniformement dans la phase et bien entoures de
  `known=1`, donc la fraction frontiere+isoles serait < 20%.
- Data slice: val seed 0, 5 vols, descent_shallow.
- Strata: par vol; aussi decomposer la fraction en
  {frontiere seule, isole seul, les deux}.

Usage
-----
    uv run python scripts/debug/diag_round2b_phase_classification.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

__all__ = ["main"]

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DELTA_PATH = Path("data/flights.delta")
REPORT_PATH = Path("data/mardown/round2b_phase_classification.md")
STEP_S = 4.0
WINDOW_S = 60.0  # +/-60s for isolation check
BOUNDARY_FRAC = 0.05  # within 5% of segment edges
SURVIVE_THRESHOLD = 0.50
DIE_THRESHOLD = 0.20

TARGET_FLIGHTS = [
    "3c6634_DLH4PV_s0",
    "0aca66_AVA8443_s0",
    "0aca66_AVA8447_s0",
    "0aca66_AVA8548_s0",
    "0aca66_AVA8558_s0",
]

# Phase thresholds (mirror dataset_regime_stats.py + diag_target_known_stratification.py)
GAMMA_CRUISE_RAD = math.radians(0.5)
GAMMA_CLIMB_RAD = math.radians(1.0)
GAMMA_DESCENT_RAD = math.radians(-1.0)

FTMIN_TO_MS = 0.00508
ALT_RATE_CRUISE = 200 * FTMIN_TO_MS
ALT_RATE_CLIMB = 500 * FTMIN_TO_MS
ALT_RATE_DESCENT = -500 * FTMIN_TO_MS
ALT_RATE_CLIMB_SAT = 1500 * FTMIN_TO_MS
ALT_RATE_DESCENT_SAT = -1000 * FTMIN_TO_MS

DURATION_BUCKETS = [
    ("<30s", 0.0, 30.0),
    ("30-120s", 30.0, 120.0),
    ("120-600s", 120.0, 600.0),
    (">600s", 600.0, float("inf")),
]


# ----------------------------------------------------------------------
# Phase classification (numpy, identical thresholds to dataset_regime_stats.py)
# ----------------------------------------------------------------------
def _classify_phases_np(gamma: np.ndarray, d_alt: np.ndarray) -> np.ndarray:
    n = len(gamma)
    labels = np.full(n, "transition", dtype=object)
    is_cruise = (np.abs(gamma) < GAMMA_CRUISE_RAD) & (np.abs(d_alt) < ALT_RATE_CRUISE)
    is_climb = (gamma > GAMMA_CLIMB_RAD) | (d_alt > ALT_RATE_CLIMB)
    is_descent = (gamma < GAMMA_DESCENT_RAD) | (d_alt < ALT_RATE_DESCENT)
    for i in range(n):
        if is_cruise[i]:
            labels[i] = "cruise"
        elif is_climb[i]:
            labels[i] = (
                "climb_saturated" if d_alt[i] > ALT_RATE_CLIMB_SAT else "climb_shallow"
            )
        elif is_descent[i]:
            labels[i] = (
                "descent_saturated"
                if d_alt[i] < ALT_RATE_DESCENT_SAT
                else "descent_shallow"
            )
    return labels


# ----------------------------------------------------------------------
# Per-flight analysis
# ----------------------------------------------------------------------
@dataclass
class FlightStats:
    flight_id: str
    n_known0_desc_shallow: int
    n_boundary: int
    n_isolated: int
    n_combined: int  # boundary OR isolated
    n_both: int  # boundary AND isolated
    segment_durations_s: list[float]


def analyze_flight(flight_df: pl.DataFrame, flight_id: str) -> FlightStats | None:
    """Compute boundary/isolation flags for descent_shallow tas_known=0 samples."""
    if flight_df.height < 5:
        return None

    flight_df = flight_df.sort("raw_timestamp")
    # Wall-clock time in seconds since flight start
    ts = flight_df["raw_timestamp"].to_numpy()
    t_s = (ts - ts[0]).astype("timedelta64[ms]").astype(np.float64) / 1000.0

    gamma = flight_df["fdm_gamma_rad"].to_numpy().astype(np.float64)
    d_alt = flight_df["fdm_d_alt_ms"].to_numpy().astype(np.float64)
    tas_known = flight_df["fdm_tas_target_known"].to_numpy().astype(bool)

    # Drop rows with non-finite gamma / d_alt (cannot classify)
    finite_mask = np.isfinite(gamma) & np.isfinite(d_alt)
    if finite_mask.sum() < 5:
        return None
    t_s = t_s[finite_mask]
    gamma = gamma[finite_mask]
    d_alt = d_alt[finite_mask]
    tas_known = tas_known[finite_mask]

    labels = _classify_phases_np(gamma, d_alt)

    # Identify contiguous descent_shallow segments (by index, since rows are sorted by time)
    is_desc_shallow = labels == "descent_shallow"
    # segments: list of (start_idx, end_idx) inclusive
    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, flag in enumerate(is_desc_shallow):
        if flag and not in_seg:
            in_seg = True
            seg_start = i
        elif not flag and in_seg:
            in_seg = False
            segments.append((seg_start, i - 1))
    if in_seg:
        segments.append((seg_start, len(is_desc_shallow) - 1))

    # Segment durations
    seg_durations = []
    for s, e in segments:
        if e == s:
            seg_durations.append(0.0)
        else:
            seg_durations.append(float(t_s[e] - t_s[s]))

    # Pre-compute timestamps of all known=1 samples (any regime) in this flight
    t_known1 = t_s[tas_known]

    # For each descent_shallow known=0 sample, compute boundary + isolated flags
    n_k0 = 0
    n_boundary = 0
    n_isolated = 0
    n_combined = 0
    n_both = 0
    for s, e in segments:
        seg_dur = t_s[e] - t_s[s] if e > s else 0.0
        for i in range(s, e + 1):
            if tas_known[i]:
                continue  # only known=0 samples
            n_k0 += 1
            # Boundary flag: relative position within segment
            if seg_dur <= 0:
                # Single-point segment: by definition both at start and end
                rel_pos = 0.0
                is_boundary = True
            else:
                rel_pos = (t_s[i] - t_s[s]) / seg_dur
                is_boundary = (rel_pos < BOUNDARY_FRAC) or (rel_pos > 1.0 - BOUNDARY_FRAC)

            # Isolated flag: any known=1 sample within +/- WINDOW_S?
            if t_known1.size == 0:
                is_isolated = True
            else:
                # find min |dt|
                dt = np.abs(t_known1 - t_s[i])
                is_isolated = bool(dt.min() > WINDOW_S)

            if is_boundary:
                n_boundary += 1
            if is_isolated:
                n_isolated += 1
            if is_boundary or is_isolated:
                n_combined += 1
            if is_boundary and is_isolated:
                n_both += 1

    return FlightStats(
        flight_id=flight_id,
        n_known0_desc_shallow=n_k0,
        n_boundary=n_boundary,
        n_isolated=n_isolated,
        n_combined=n_combined,
        n_both=n_both,
        segment_durations_s=seg_durations,
    )


# ----------------------------------------------------------------------
# Aggregation + rendering
# ----------------------------------------------------------------------
def _pct(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{100.0 * num / den:.2f}%"


def _bucketize(durations: list[float]) -> dict[str, int]:
    counts = {label: 0 for label, _, _ in DURATION_BUCKETS}
    for d in durations:
        for label, lo, hi in DURATION_BUCKETS:
            if lo <= d < hi:
                counts[label] += 1
                break
    return counts


def render_markdown(per_flight: list[FlightStats]) -> tuple[str, dict]:
    # Aggregate
    agg_n = sum(f.n_known0_desc_shallow for f in per_flight)
    agg_bnd = sum(f.n_boundary for f in per_flight)
    agg_iso = sum(f.n_isolated for f in per_flight)
    agg_comb = sum(f.n_combined for f in per_flight)
    agg_both = sum(f.n_both for f in per_flight)

    all_durations: list[float] = []
    for f in per_flight:
        all_durations.extend(f.segment_durations_s)
    median_dur = float(np.median(all_durations)) if all_durations else float("nan")

    bucket_counts = _bucketize(all_durations)

    frac_combined = (agg_comb / agg_n) if agg_n > 0 else float("nan")

    lines: list[str] = []
    lines.append("# Round 2B -- Phase classification / boundary proximity")
    lines.append("")
    lines.append("## Hypothesis under test")
    lines.append("")
    lines.append(
        "Les samples descent_shallow `tas_known=0` sont concentres aux frontieres de"
        " phase (debut/fin de la descente) ou en zones isolees (long ecart temporel"
        " depuis le dernier `tas_known=1`), ou le contexte dynamique est mal defini"
        " -- la classification descent_shallow est partiellement erronee pour ces"
        " samples, et le bias +19.73 m/s reflete ce mismatch de contexte."
    )
    lines.append("")
    lines.append("## Falsifier")
    lines.append("")
    lines.append(
        "- **Statistique primaire**: fraction des samples `known=0` descent_shallow"
        " qui sont SOIT (a) a moins de 5% de la frontiere temporelle de la phase de"
        " descente (au debut ou a la fin), SOIT (b) \"isoles\" -- pas de sample"
        " `known=1` dans une fenetre de +/-60s (15 samples a 4s)."
    )
    lines.append(
        "- **Threshold**: H_2B survives si fraction > 50% sur l'agregat."
        " Dies si < 20%. Indetermine entre."
    )
    lines.append(
        "- **Null world**: si H_2B etait fausse, les samples `known=0` descent_shallow"
        " seraient distribues uniformement dans la phase et bien entoures de `known=1`,"
        " donc la fraction frontiere+isoles serait < 20%."
    )
    lines.append("- **Data slice**: val seed 0, 5 vols, descent_shallow.")
    lines.append(
        "- **Strata**: par vol; aussi decomposer la fraction en"
        " {frontiere seule, isole seul, les deux}."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Pour chaque vol: tri par `raw_timestamp`, classification phase via"
        " (gamma, d_alt) avec memes seuils que `dataset_regime_stats.py`."
        " Identification des segments contigus descent_shallow par scan d'indices."
        " Pour chaque sample `tas_known=0` du segment: `flag_boundary` = position"
        " relative <0.05 ou >0.95 (segments d'un seul point => boundary par defaut);"
        " `flag_isolated` = aucun sample `tas_known=1` (toutes phases confondues, meme"
        " vol) dans une fenetre +/-60s en wall-clock. Lecture single-pass de"
        " `data/flights.delta`, filtrage `meta_split == 'val'` puis sur les 5 vols cibles."
    )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append("### Aggregate (5 flights)")
    lines.append("")
    lines.append(
        "| n_known=0 desc_shallow | % boundary | % isolated | % boundary OR isolated |"
        " % boundary AND isolated | median segment duration (s) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {agg_n} | {_pct(agg_bnd, agg_n)} | {_pct(agg_iso, agg_n)} |"
        f" {_pct(agg_comb, agg_n)} | {_pct(agg_both, agg_n)} |"
        f" {median_dur:.1f} |"
    )
    lines.append("")
    lines.append("### Per-flight")
    lines.append("")
    lines.append(
        "| Flight | n | % boundary | % isolated | % combined | % both | median seg dur (s) | n_segments |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for f in per_flight:
        med = float(np.median(f.segment_durations_s)) if f.segment_durations_s else float("nan")
        lines.append(
            f"| {f.flight_id} | {f.n_known0_desc_shallow} |"
            f" {_pct(f.n_boundary, f.n_known0_desc_shallow)} |"
            f" {_pct(f.n_isolated, f.n_known0_desc_shallow)} |"
            f" {_pct(f.n_combined, f.n_known0_desc_shallow)} |"
            f" {_pct(f.n_both, f.n_known0_desc_shallow)} |"
            f" {med:.1f} | {len(f.segment_durations_s)} |"
        )
    lines.append("")
    lines.append("### Distribution of segment durations")
    lines.append("")
    lines.append("| Bucket | n_segments |")
    lines.append("|---|---:|")
    for label, _, _ in DURATION_BUCKETS:
        lines.append(f"| {label} | {bucket_counts[label]} |")
    lines.append(f"| **total** | **{len(all_durations)}** |")
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    if agg_n == 0:
        verdict_line = (
            "H_2B **indeterminate**: zero `known=0` descent_shallow samples across"
            " the 5 flights -- cannot evaluate."
        )
    elif frac_combined > SURVIVE_THRESHOLD:
        verdict_line = (
            f"H_2B **survives**: fraction boundary OR isolated = {100.0 * frac_combined:.2f}%"
            f" > 50% on aggregate (n={agg_n})."
        )
    elif frac_combined < DIE_THRESHOLD:
        verdict_line = (
            f"H_2B **dies**: fraction boundary OR isolated = {100.0 * frac_combined:.2f}%"
            f" < 20% on aggregate (n={agg_n})."
        )
    else:
        verdict_line = (
            f"H_2B **indeterminate**: fraction boundary OR isolated ="
            f" {100.0 * frac_combined:.2f}% in [20%, 50%] on aggregate (n={agg_n})."
        )
    lines.append(verdict_line)
    lines.append("")

    # Caveats
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **Multiple disjoint descent_shallow segments**: chaque vol peut avoir"
        " plusieurs segments descent_shallow (interruptions transition / cruise"
        " transitoires). On les traite independamment et la position relative est"
        " calculee par segment, pas globalement -- un sample peut etre frontiere"
        " d'un petit segment court tout en etant loin du debut de la descente"
        " globale du vol."
    )
    lines.append(
        "- **Single-point segments**: un segment d'un seul point (ou de duree 0s) est"
        " marque boundary par definition (rel_pos = 0.0). Cela gonfle legerement la"
        " statistique boundary si le bruit de classification produit des segments"
        " degeneres -- a croiser avec la table de duree des segments."
    )
    lines.append(
        f"- **Fenetre +/-{WINDOW_S:.0f}s**: choisie pour valoir 15 samples a {STEP_S:.0f}s"
        " (cf. CLAUDE.md / target_known_diagnosis.md). Une fenetre plus large reduit la"
        " fraction d'isoles; plus etroite l'augmente. Sensibilite non testee ici."
    )
    lines.append(
        "- **Sampling rate**: 4s par sample (cf. `STEP_S` dans"
        " `diag_target_known_stratification.py` et CLAUDE.md). On utilise les"
        " timestamps wall-clock, donc une eventuelle non-uniformite (gaps reproduits"
        " a 4s par le preprocessing) est ignoree -- les distances temporelles sont"
        " mesurees sur `raw_timestamp` directement."
    )
    lines.append(
        "- **\"isolated\" tient compte de toutes phases**: la fenetre +/-60s scanne"
        " tous les samples `tas_known=1` du vol independamment du regime, comme"
        " demande dans la mission."
    )
    lines.append("")

    summary = {
        "agg_n": agg_n,
        "agg_boundary": agg_bnd,
        "agg_isolated": agg_iso,
        "agg_combined": agg_comb,
        "agg_both": agg_both,
        "frac_combined": frac_combined,
        "median_seg_dur_s": median_dur,
    }
    return "\n".join(lines), summary


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> int:
    # Echo falsifier header (mission requirement)
    print("=" * 72)
    print("Round 2B falsifier")
    print("-" * 72)
    print("Statistic : fraction(known=0 desc_shallow that are boundary OR isolated)")
    print("Threshold : survives > 50%, dies < 20%, indeterminate in [20%, 50%]")
    print("Null world: uniform distribution => fraction < 20%")
    print("Data slice: val seed 0, 5 flights, descent_shallow")
    print("Strata    : per-flight + decomposition {boundary only, isolated only, both}")
    print("=" * 72)
    print()

    if not DELTA_PATH.exists():
        print(f"ERROR: delta path not found at {DELTA_PATH}", file=sys.stderr)
        return 1

    print(f"[load] reading {DELTA_PATH} (val split, target flights only)...")
    df = (
        pl.scan_delta(str(DELTA_PATH))
        .filter(pl.col("meta_split") == "val")
        .filter(pl.col("meta_flight_id").is_in(TARGET_FLIGHTS))
        .filter(pl.col("fdm_flag_valid"))
        .select(
            [
                "meta_flight_id",
                "raw_timestamp",
                "fdm_gamma_rad",
                "fdm_d_alt_ms",
                "fdm_tas_target_known",
            ]
        )
        .collect()
    )
    print(f"[load] got {df.height:,} rows across {df['meta_flight_id'].n_unique()} flights")

    per_flight: list[FlightStats] = []
    for fid in TARGET_FLIGHTS:
        flight_df = df.filter(pl.col("meta_flight_id") == fid)
        if flight_df.height == 0:
            print(f"[warn] flight {fid} not found in val split (skipping)")
            per_flight.append(
                FlightStats(
                    flight_id=fid,
                    n_known0_desc_shallow=0,
                    n_boundary=0,
                    n_isolated=0,
                    n_combined=0,
                    n_both=0,
                    segment_durations_s=[],
                )
            )
            continue
        stats = analyze_flight(flight_df, fid)
        if stats is None:
            print(f"[warn] flight {fid} too short / no finite phase samples (skipping)")
            per_flight.append(
                FlightStats(
                    flight_id=fid,
                    n_known0_desc_shallow=0,
                    n_boundary=0,
                    n_isolated=0,
                    n_combined=0,
                    n_both=0,
                    segment_durations_s=[],
                )
            )
            continue
        per_flight.append(stats)
        print(
            f"[ok] {fid}: n_k0_desc_shallow={stats.n_known0_desc_shallow},"
            f" boundary={stats.n_boundary}, isolated={stats.n_isolated},"
            f" combined={stats.n_combined}, n_segments={len(stats.segment_durations_s)}"
        )

    md, summary = render_markdown(per_flight)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md)
    print()
    print("Aggregate summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    print(f"Report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
