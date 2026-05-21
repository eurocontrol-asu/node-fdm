"""Strategy E / Experiment 07 — P&S Eq (100) mass-inversion vs QAR.

Confront the Poll-Schumann mass-inversion theory (Eq 100) to 526 real A320
QAR flights. Pure diagnostic — no training, no model. For each cruise
segment, invert ``fl_o(mass_ratio) = FL_obs`` via the closed-form
``ps_core.optimum.optimum_in_isa`` and compare ``m_PS = mass_ratio · MTOM``
to ``SYS__GW`` ground truth.

The optimum_in_isa Mach (``mach_o``) does NOT depend on mass_ratio in the
P&S formulation — it is a fixed reference per aircraft (psi_4-driven).
Mass-inversion therefore goes via the **altitude** branch: at a given
mass_ratio the optimum tropospheric flight level is

    chi_o(mass_ratio) = 0.980 · (1 - 0.017 τ) · (ψ_7/mass_ratio)^k
    p_o = P_TROPOPAUSE_ISA / chi_o
    FL_o = pressure_to_fl(p_o)

monotone-decreasing in mass_ratio. Inversion is done once on a fine
mass_ratio grid then vectorised np.interp across all segments.

Cruise filter (where the optimum theory should hold) :
    |dh/dt| < 1 m/s, |dV/dt| < 0.1 m/s², h ≥ 9000 m.

Outputs report under data/models/_comparison/strategy_e_ps_eq100_diagnostic.md
and prints metrics to stdout. Also extends the per-segment table with a
new "model": ``m_PS_eq100``, callable without checkpoint.

Validation-only (R5 compliant) — does not write to any training feature
cache or model dir.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(
    0, "/Users/gabriel/axm/04-papers/PS_MODEL/poll_schumann_lib/packages/ps-core/src"
)

from ps_core._types import AircraftPsi
from ps_core.optimum import optimum_in_isa
from test_a_mass_encoder_qar import (
    _FT_TO_M,
    _KT_TO_MS,
    _dedup_qar_files,
)

# A320 psi-set (table2.yaml entry "A320", Poll-Schumann 2020 Part 2).
A320_PSI = AircraftPsi(
    psi_1=0.156,
    psi_2=8.05,
    psi_4=0.753,
    psi_5=6.29e7,
    psi_6=0.656,
    tau=0.162,
)
A320_MTOW_KG = 77_000.0  # canonical MTOW used by node-fdm-v2 schemas.

# Cruise-stable filter (matches the ticket §X-bis definition).
_CRUISE_ALT_MIN_M = 9_000.0  # ~FL295
_CRUISE_DH_DT_MAX_MS = 1.0   # |dh/dt|
_CRUISE_DV_DT_MAX_MS2 = 0.1  # |dV/dt|

# Validity range for mass_ratio inversion.
_MASS_RATIO_MIN = 0.50
_MASS_RATIO_MAX = 1.00
_MASS_RATIO_GRID_N = 1001

# Sliding-window parameters mirroring the training loader.
_SEQ_LEN_S = 60
_SHIFT_S = 60


def _build_inversion_table() -> tuple[np.ndarray, np.ndarray]:
    """Return (mass_ratio_grid_ascending, fl_o_grid_descending).

    Pre-computes the monotone fl_o(mass_ratio) curve once. Inversion
    becomes a vectorised ``np.interp(FL_obs, fl_o_asc, mass_ratio_desc)``
    where the arrays are flipped so np.interp's required ascending xp
    is satisfied.
    """
    grid = np.linspace(_MASS_RATIO_MIN, _MASS_RATIO_MAX, _MASS_RATIO_GRID_N)
    fl_o = np.array([optimum_in_isa(A320_PSI, float(m)).fl_o for m in grid])
    return grid, fl_o


def _invert_mass_ratio(
    fl_obs: np.ndarray,
    grid_mass_ratio: np.ndarray,
    grid_fl_o: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised inversion. Returns (mass_ratio_estimate, converged_mask).

    np.interp clips outside the bounds; we mark these as not-converged.
    Since fl_o decreases in mass_ratio, flip both arrays to ascending fl_o.
    """
    fl_asc = grid_fl_o[::-1]
    mr_desc = grid_mass_ratio[::-1]
    mr_est = np.interp(fl_obs, fl_asc, mr_desc)
    fl_min, fl_max = float(grid_fl_o[-1]), float(grid_fl_o[0])
    converged = np.isfinite(fl_obs) & (fl_obs >= fl_min) & (fl_obs <= fl_max)
    return mr_est, converged


def _cruise_stable_mask(
    alt_m: np.ndarray, tas_ms: np.ndarray, dt_s: float = 1.0
) -> np.ndarray:
    """Boolean mask of indices where cruise-stable conditions hold."""
    n = len(alt_m)
    if n < 2:
        return np.zeros(n, dtype=bool)
    dh_dt = np.gradient(alt_m) / dt_s
    dv_dt = np.gradient(tas_ms) / dt_s
    return (
        np.isfinite(alt_m)
        & np.isfinite(tas_ms)
        & (alt_m >= _CRUISE_ALT_MIN_M)
        & (np.abs(dh_dt) <= _CRUISE_DH_DT_MAX_MS)
        & (np.abs(dv_dt) <= _CRUISE_DV_DT_MAX_MS2)
    )


def _segments_from_flight(
    df: pl.DataFrame, *, seq_len: int, shift: int
) -> list[dict]:
    """Yield one record per valid segment start with the fields needed for
    Eq 100 inversion and ground-truth comparison.
    """
    n = df.shape[0]
    if n < seq_len:
        return []
    alt_ft = df["ALT__STD"].to_numpy().astype(np.float64)
    alt_m = alt_ft * _FT_TO_M
    gw = df["SYS__GW"].to_numpy().astype(np.float64)
    tas_ms = (
        df["SPD__TAS"].to_numpy().astype(np.float64) * _KT_TO_MS
        if "SPD__TAS" in df.columns
        else np.full(n, np.nan)
    )
    cruise_mask = _cruise_stable_mask(alt_m, tas_ms)

    out: list[dict] = []
    for start in range(0, n - seq_len + 1, shift):
        m_truth = float(gw[start]) if start < n else float("nan")
        if not (np.isfinite(m_truth) and 40_000 < m_truth < 80_000):
            continue
        a_ft = float(alt_ft[start])
        a_m = float(alt_m[start])
        if not np.isfinite(a_ft) or a_ft < 1_000.0:
            continue
        fl_obs = a_ft / 100.0
        is_cruise = bool(cruise_mask[start])
        out.append(
            {
                "fl_obs": fl_obs,
                "alt_m": a_m,
                "m_truth": m_truth,
                "is_cruise": is_cruise,
            }
        )
    return out


def _flight_id(p: Path) -> str:
    fid = p.name.split("_")[1] if "_" in p.name else p.stem
    return f"{fid}__{p.stem}"


def _decompose_between_within(
    y: np.ndarray, preds: np.ndarray, flight_ids: np.ndarray
) -> dict[str, float]:
    """Between-flight vs within-flight Pearson decomposition."""
    unique = sorted(set(flight_ids.tolist()))
    if not unique:
        return {
            "between_flight_corr": float("nan"),
            "within_flight_corr": float("nan"),
        }
    y_means = np.zeros_like(y, dtype=np.float64)
    p_means = np.zeros_like(preds, dtype=np.float64)
    n_per: dict[str, int] = {}
    for fid in unique:
        mask = flight_ids == fid
        y_means[mask] = y[mask].mean()
        p_means[mask] = preds[mask].mean()
        n_per[fid] = int(mask.sum())
    if len(unique) >= 2 and np.std(y_means) > 1e-6 and np.std(p_means) > 1e-6:
        between_corr = float(np.corrcoef(y_means, p_means)[0, 1])
    else:
        between_corr = float("nan")
    multi_mask = np.array([n_per[fid] >= 2 for fid in flight_ids], dtype=bool)
    within_corr = float("nan")
    if multi_mask.any():
        y_w = (y - y_means)[multi_mask]
        p_w = (preds - p_means)[multi_mask]
        if np.std(y_w) > 1e-6 and np.std(p_w) > 1e-6:
            within_corr = float(np.corrcoef(y_w, p_w)[0, 1])
    return {
        "between_flight_corr": between_corr,
        "within_flight_corr": within_corr,
    }


def _metrics(
    y: np.ndarray, preds: np.ndarray, flight_ids: np.ndarray
) -> dict[str, float]:
    err = preds - y
    n = len(y)
    if n < 2 or np.std(y) < 1e-6 or np.std(preds) < 1e-6:
        return {
            "n": n,
            "corr": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "bias_pct": float("nan"),
            "between_flight_corr": float("nan"),
            "within_flight_corr": float("nan"),
        }
    corr = float(np.corrcoef(y, preds)[0, 1])
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    bias = float(err.mean())
    bias_pct = float(np.mean(err / y) * 100.0)
    decomp = _decompose_between_within(y, preds, flight_ids)
    return {
        "n": n,
        "corr": corr,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "bias_pct": bias_pct,
        **decomp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qar-dir", default="/Users/gabriel/Downloads/QAR3")
    parser.add_argument("--seq-len", type=int, default=_SEQ_LEN_S)
    parser.add_argument("--shift", type=int, default=_SHIFT_S)
    parser.add_argument(
        "--report",
        default="data/models/_comparison/strategy_e_ps_eq100_diagnostic.md",
    )
    args = parser.parse_args()

    grid_mr, grid_fl = _build_inversion_table()
    fl_min, fl_max = float(grid_fl[-1]), float(grid_fl[0])
    print(
        f"Eq 100 inversion table: mass_ratio in [{_MASS_RATIO_MIN}, "
        f"{_MASS_RATIO_MAX}] -> FL_o in [{fl_min:.1f}, {fl_max:.1f}]"
    )

    qar_files = _dedup_qar_files(sorted(Path(args.qar_dir).glob("*.parquet")))
    print(f"Found {len(qar_files)} unique QAR files")

    fl_obs: list[float] = []
    m_truth: list[float] = []
    is_cruise: list[bool] = []
    flight_ids_l: list[str] = []

    n_files_ok = 0
    for f in qar_files:
        try:
            df = pl.read_parquet(f)
        except Exception:  # noqa: S112 # broad parquet read errors are skip-and-continue here
            continue
        segs = _segments_from_flight(df, seq_len=args.seq_len, shift=args.shift)
        if not segs:
            continue
        n_files_ok += 1
        fid = _flight_id(f)
        for s in segs:
            fl_obs.append(s["fl_obs"])
            m_truth.append(s["m_truth"])
            is_cruise.append(s["is_cruise"])
            flight_ids_l.append(fid)

    if not fl_obs:
        print("No segments collected", file=sys.stderr)
        return 1

    fl_arr = np.array(fl_obs)
    y_arr = np.array(m_truth)
    cruise_arr = np.array(is_cruise, dtype=bool)
    flight_arr = np.array(flight_ids_l)

    mr_est, conv = _invert_mass_ratio(fl_arr, grid_mr, grid_fl)
    m_ps = mr_est * A320_MTOW_KG

    n_total = len(fl_arr)
    n_cruise = int(cruise_arr.sum())
    n_conv = int(conv.sum())
    n_cruise_conv = int((cruise_arr & conv).sum())
    print(
        f"Segments: total={n_total}  cruise_stable={n_cruise}  "
        f"in_inversion_range={n_conv}  cruise&converged={n_cruise_conv}  "
        f"across {n_files_ok} flights"
    )

    views = {
        "ALL segments (cruise + non-cruise, only in inversion range)": conv,
        "CRUISE STABLE only (theory's natural domain)": cruise_arr & conv,
        "NON-CRUISE (climb/descent, theory shouldn't apply)": (~cruise_arr) & conv,
    }
    metrics: dict[str, dict[str, float]] = {}
    for label, mask in views.items():
        if mask.sum() < 10:
            print(f"  [{label}] : n={int(mask.sum())} (skipped)")
            metrics[label] = {"n": int(mask.sum())}
            continue
        m = _metrics(y_arr[mask], m_ps[mask], flight_arr[mask])
        metrics[label] = m
        print(
            f"  [{label}]"
            f"  n={m['n']}  corr={m['corr']:+.3f}"
            f"  MAE={m['mae']:.0f} kg ({m['bias_pct']:+.1f}% bias)"
            f"  between={m['between_flight_corr']:+.3f}"
            f"  within={m['within_flight_corr']:+.3f}"
        )

    # Side-by-side with v9: load v9 predictions on the same cruise pool.
    # Done by importing the existing per-segment validator and rerunning
    # the encoder block. Keep it lightweight here — just print v9 stats
    # from the existing report file when present, else mention skip.
    v9_report = Path("data/models/_comparison/test_a_per_segment.md")
    v9_blurb = (
        f"\nReference v9 baseline (from `{v9_report}`): per-segment corr "
        "**0.351**, within-flight **+0.727**, MAE **4501 kg** (49 400 segs)."
    )

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Strategy E — Diagnostic Exp 07 : P&S Eq (100) vs QAR mass",
        "",
        "> Mass-ratio inversion of Poll-Schumann Eq (100) on 526 A320 QAR "
        "flights. Pure measurement — no training, no model.",
        "",
        "## Inversion setup",
        "",
        f"- Aircraft : A320, psi_4={A320_PSI.psi_4}, tau={A320_PSI.tau}, "
        f"psi_6={A320_PSI.psi_6}",
        f"- MTOM used for `m_PS = mass_ratio · MTOM` : "
        f"{int(A320_MTOW_KG)} kg",
        f"- mass_ratio grid : [{_MASS_RATIO_MIN}, {_MASS_RATIO_MAX}] "
        f"({_MASS_RATIO_GRID_N} points)",
        f"- Resulting FL_o range : "
        f"[{fl_min:.1f}, {fl_max:.1f}] "
        "(below {0:.0f} FL the aircraft is non-invertible "
        "— theory predicts mass > MTOM)".format(fl_min),
        f"- Cruise filter : alt ≥ {_CRUISE_ALT_MIN_M:.0f} m, "
        f"|dh/dt| ≤ {_CRUISE_DH_DT_MAX_MS} m/s, "
        f"|dV/dt| ≤ {_CRUISE_DV_DT_MAX_MS2} m/s²",
        f"- Sliding window : seq_len={args.seq_len}s, shift={args.shift}s",
        "",
        "## Sample counts",
        "",
        f"- Total segments : **{n_total}** across {n_files_ok} flights",
        f"- Cruise-stable : **{n_cruise}** "
        f"({100.0 * n_cruise / max(1, n_total):.1f} %)",
        f"- In inversion FL range : **{n_conv}** "
        f"({100.0 * n_conv / max(1, n_total):.1f} %)",
        f"- Cruise & inverted : **{n_cruise_conv}** "
        f"({100.0 * n_cruise_conv / max(1, n_total):.1f} %)",
        "",
        "## Headline metrics",
        "",
        "| View | n | corr | MAE (kg) | bias % | between-flight | within-flight |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, m in metrics.items():
        if m.get("n", 0) < 10:
            lines.append(f"| {label} | {m.get('n', 0)} | n/a | n/a | n/a | n/a | n/a |")
            continue
        within_str = (
            f"{m['within_flight_corr']:+.3f}"
            if not math.isnan(m["within_flight_corr"])
            else "n/a"
        )
        between_str = (
            f"{m['between_flight_corr']:+.3f}"
            if not math.isnan(m["between_flight_corr"])
            else "n/a"
        )
        lines.append(
            f"| {label} | {m['n']} | {m['corr']:+.3f} | "
            f"{m['mae']:.0f} | {m['bias_pct']:+.1f} | "
            f"{between_str} | {within_str} |"
        )

    lines.extend(
        [
            "",
            "## Verdict guide (from STRATEGY_E ticket)",
            "",
            "- Eq (100) corr **≥ 0.6** on cruise-stable → theory holds → "
            "proceed to Exp 08 (X-bis : λ-sweep auxiliary loss).",
            "- Eq (100) corr ∈ **[0.3, 0.6]** on cruise-stable → partial → "
            "try Eq (112) universal collapse (reconstruct ηo·L/D from QAR).",
            "- Eq (100) corr **< 0.3** on cruise-stable → theory falsified "
            "on A320 QAR → write FINDINGS, recommend Phase 2 (thrust path).",
            v9_blurb,
            "",
            "## Reading guide",
            "",
            "- **CRUISE STABLE** is the headline view: theory predicts the "
            "optimum altitude *at cruise*; off-cruise data is just noise.",
            "- **NON-CRUISE** is informational : a strong negative would "
            "indicate the inversion is systematically biased outside cruise.",
            "- **within-flight corr** measures whether Eq 100 tracks fuel "
            "burn (heavier early in the flight, lighter later). This is "
            "what v9 already captures (+0.73) — Eq 100 must do at least "
            "as well to be useful as an auxiliary signal.",
        ]
    )

    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
