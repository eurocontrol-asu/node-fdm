"""Lateral trajectory computations for Neural ODE control inputs.

Detects straight-line segments between turning points, then computes
orthodromic (great-circle) and loxodromic (rhumb-line) reference tracks
for each segment.  These serve as **lateral control inputs** for the
flight dynamics model: the Neural ODE learns aircraft response to
these navigation commands, analogous to how ``fdm_mach_sel`` / ``fdm_vz_sel_ftmin``
serve as longitudinal commands.

Key outputs:

- ``track_ortho`` -- great-circle bearing from segment start A to end B
- ``track_loxo``  -- rhumb-line (constant heading) bearing A -> B
- ``drift_angle``  -- heading - track (crosswind effect)
- ``lat_wind``    -- lateral wind component: TAS x sin(drift_angle)
- ``in_turn``     -- boolean mask for turning vs straight flight

Ports and improves logic from the legacy ``opensky_v2`` branch:
``lateral_computations.py`` and ``15_curvature.py``.

Example::

    from node_fdm_data.lateral import augment_lateral

    result = augment_lateral(flight_df)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

__all__ = [
    "augment_lateral",
    "detect_turning_points",
    "orthodromic_bearing",
    "rhumb_bearing",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R_EARTH_M: float = 6_371_000.0
"""Mean Earth radius in metres."""

_MIN_POINTS: int = 4
"""Minimum points needed for turn detection (diff(4) needs at least 5)."""


# ---------------------------------------------------------------------------
# Low-level geometry (vectorised, radians in / radians out)
# ---------------------------------------------------------------------------


def orthodromic_bearing(
    phi1: npt.ArrayLike,
    lam1: npt.ArrayLike,
    phi2: npt.ArrayLike,
    lam2: npt.ArrayLike,
) -> np.ndarray:
    """Initial great-circle bearing from (φ₁, λ₁) to (φ₂, λ₂).

    All inputs and output in **radians**.  Result is in ``[0, 2pi)``.
    """
    phi1, lam1 = np.asarray(phi1), np.asarray(lam1)
    phi2, lam2 = np.asarray(phi2), np.asarray(lam2)
    d_lam = lam2 - lam1
    y = np.sin(d_lam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lam)
    bearing = np.arctan2(y, x)
    return np.asarray((bearing + 2 * np.pi) % (2 * np.pi))


def rhumb_bearing(
    phi1: npt.ArrayLike,
    lam1: npt.ArrayLike,
    phi2: npt.ArrayLike,
    lam2: npt.ArrayLike,
) -> np.ndarray:
    """Rhumb-line (loxodromic) bearing from (ph1, la1) to (ph2, la2).

    All inputs and output in **radians**.  Result is in ``[0, 2pi)``.
    Handles degenerate cases (same latitude, same point).
    """
    eps = 1e-12
    phi1, lam1 = np.asarray(phi1), np.asarray(lam1)
    phi2, lam2 = np.asarray(phi2), np.asarray(lam2)

    d_lam = lam2 - lam1
    d_psi = np.log(
        np.maximum(np.tan(phi2 / 2 + np.pi / 4), eps)
        / np.maximum(np.tan(phi1 / 2 + np.pi / 4), eps)
    )

    bearing = np.arctan2(d_lam, d_psi)

    # Degenerate: Δψ ≈ 0 (same latitude)
    degen_psi = np.abs(d_psi) < eps
    degen_both = degen_psi & (np.abs(d_lam) < eps)
    ew = np.where(d_lam > 0, np.pi / 2, -np.pi / 2)

    bearing = np.where(degen_both, np.nan, bearing)
    bearing = np.where(degen_psi & ~degen_both, ew, bearing)

    return (bearing + 2 * np.pi) % (2 * np.pi)


# ---------------------------------------------------------------------------
# Turn detection (from legacy 15_curvature.py — smoothed angular rate)
# ---------------------------------------------------------------------------


def detect_turning_points(
    track_deg: npt.NDArray[np.floating[Any]],
    *,
    diff_n: int = 4,
    dt: float = 4.0,
    threshold_deg_per_sec: float = 0.05,
    min_straight_len: int = 10,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
    """Detect turns using smoothed angular rate thresholding.

    Uses ``|track.diff(n) / (n x dt)| >= threshold`` to identify
    turning points.  This is the robust approach from the legacy
    ``15_curvature.py`` script.

    Args:
        track_deg: Track angle in degrees (may wrap at 0/360).
        diff_n: Number of points for finite differencing.
        dt: Sampling interval in seconds.
        threshold_deg_per_sec: Turn rate threshold (°/s).
        min_straight_len: Minimum length to keep a straight segment.

    Returns:
        ``(segment_indices, in_turn)`` where *segment_indices* are
        the start-of-segment indices and *in_turn* is a boolean mask.
    """
    n = len(track_deg)
    if n < _MIN_POINTS + 1:
        return np.array([0], dtype=np.intp), np.zeros(n, dtype=np.bool_)

    # Unwrap to avoid 0/360 jumps
    track_uw = np.degrees(np.unwrap(np.radians(track_deg)))

    # Smoothed angular rate (°/s): |Δtrack / Δt|
    d_track = np.abs(np.diff(track_uw, n=diff_n, prepend=[track_uw[0]] * diff_n))
    rate = d_track / (diff_n * dt)

    in_turn = rate >= threshold_deg_per_sec
    _fill_short_straight_gaps(in_turn, min_straight_len)
    boundaries = _segment_boundaries(in_turn)

    return boundaries, in_turn


def _fill_short_straight_gaps(in_turn: npt.NDArray[np.bool_], min_len: int) -> None:
    """Mark short straight runs (< ``min_len``) as turning, in-place.

    Matches the legacy behavior: a gap is filled only when entering a turn,
    i.e. it must be bounded by turns on both sides.
    """
    straight = ~in_turn
    edges = np.diff(straight.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    n = in_turn.size
    fillable = (ends - starts < min_len) & (starts > 0) & (ends < n)
    for s, e in zip(starts[fillable], ends[fillable], strict=True):
        in_turn[s:e] = True


def _segment_boundaries(in_turn: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
    """Indices of turn→straight transitions, prefixed with 0."""
    transitions = np.flatnonzero(in_turn[:-1] & ~in_turn[1:]) + 1
    return np.concatenate(([0], transitions)).astype(np.intp)


# ---------------------------------------------------------------------------
# Segment endpoint assignment
# ---------------------------------------------------------------------------


def _assign_segment_endpoints(
    lat: np.ndarray,
    lon: np.ndarray,
    seg_indices: np.ndarray,
    in_turn: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map each point to its segment's start (A) and end (B) coords."""
    lat_a = np.full(n, np.nan)
    lon_a = np.full(n, np.nan)
    lat_b = np.full(n, np.nan)
    lon_b = np.full(n, np.nan)

    for k in range(len(seg_indices)):
        start = seg_indices[k]
        end = seg_indices[k + 1] - 1 if k + 1 < len(seg_indices) else n - 1

        # Only assign for straight segments
        for i in range(start, end + 1):
            if not in_turn[i]:
                lat_a[i] = lat[start]
                lon_a[i] = lon[start]
                lat_b[i] = lat[end]
                lon_b[i] = lon[end]

    return lat_a, lon_a, lat_b, lon_b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def augment_lateral(
    df: pl.DataFrame,
    *,
    dt: float = 4.0,
    threshold_deg_per_sec: float = 0.05,
    min_straight_len: int = 10,
) -> pl.DataFrame:
    """Augment a flight DataFrame with lateral reference tracks.

    Adds the following columns:

    - ``in_turn`` — bool, ``True`` during turns
    - ``track_ortho`` — orthodromic reference track (deg), NaN in turns
    - ``track_loxo`` — loxodromic reference track (deg), NaN in turns
    - ``drift_angle`` -- heading - track (deg, signed [-180, 180])
    - ``lat_wind`` -- lateral wind component (kt, positive = from left)

    Args:
        df: Single-flight eager DataFrame with ``latitude``,
            ``longitude``, ``track``, and optionally ``heading``
            and a TAS column (``era_tas_kt`` or ``TAS``).
        dt: Sampling interval (seconds).
        threshold_deg_per_sec: Turn detection threshold (°/s).
        min_straight_len: Minimum straight segment length (points).

    Returns:
        DataFrame with lateral columns added.
    """
    n = len(df)
    if n < _MIN_POINTS + 1:
        return df.with_columns(
            pl.lit(False).alias("in_turn"),
            pl.lit(None, dtype=pl.Float64).alias("track_ortho"),
            pl.lit(None, dtype=pl.Float64).alias("track_loxo"),
            pl.lit(None, dtype=pl.Float64).alias("drift_angle"),
            pl.lit(None, dtype=pl.Float64).alias("lat_wind"),
        )

    # --- Turn detection ---
    track_raw = df["track"].to_numpy().astype(np.float64)
    seg_indices, in_turn = detect_turning_points(
        track_raw,
        dt=dt,
        threshold_deg_per_sec=threshold_deg_per_sec,
        min_straight_len=min_straight_len,
    )

    # --- Segment endpoints ---
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    lat_a, lon_a, lat_b, lon_b = _assign_segment_endpoints(
        lat,
        lon,
        seg_indices,
        in_turn,
        n,
    )

    # Convert to radians for bearing computation
    phi_a, lam_a = np.radians(lat_a), np.radians(lon_a)
    phi_b, lam_b = np.radians(lat_b), np.radians(lon_b)

    # --- Orthodromic bearing (great circle) A → B ---
    # Compute from current position to B for evolving reference
    phi_cur = np.radians(lat)
    lam_cur = np.radians(lon)
    ortho_rad = orthodromic_bearing(phi_cur, lam_cur, phi_b, lam_b)
    ortho_deg = np.degrees(ortho_rad)
    ortho_deg[in_turn] = np.nan  # No reference during turns

    # --- Loxodromic bearing (rhumb line) A → B ---
    loxo_rad = rhumb_bearing(phi_a, lam_a, phi_b, lam_b)
    loxo_deg = np.degrees(loxo_rad)
    loxo_deg[in_turn] = np.nan

    # --- Drift angle and lateral wind ---
    hdg_col = "heading" if "heading" in df.columns else None
    tas_col = next(
        (c for c in ("era_tas_kt", "TAS") if c in df.columns),
        None,
    )

    if hdg_col is not None and tas_col is not None:
        hdg = df[hdg_col].to_numpy().astype(np.float64)
        tas = df[tas_col].to_numpy().astype(np.float64)
        drift = (hdg - track_raw + 180) % 360 - 180
        lat_wind = tas * np.sin(np.radians(drift))
    else:
        drift = np.full(n, np.nan)
        lat_wind = np.full(n, np.nan)

    return df.with_columns(
        pl.Series("in_turn", in_turn),
        pl.Series("track_ortho", ortho_deg),
        pl.Series("track_loxo", loxo_deg),
        pl.Series("drift_angle", drift),
        pl.Series("lat_wind", lat_wind),
    )
