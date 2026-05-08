"""Mode labelling stage — pipeline v3.

Why
---
The Cui 2019 trajectory-prediction loss is weighted per *mode* — a
categorical label that captures, for every sample, which axes are
actively controlled by the FMS.  This module assigns each sample one
of 13 mutually exclusive labels:

    TURN
    ALT_MACH    ALT_CAS    ALT_UNK
    VZ_MACH     VZ_CAS     VZ_UNK
    GAMMA_MACH  GAMMA_CAS  GAMMA_UNK
    UNKVERT_MACH  UNKVERT_CAS  UNKVERT_UNK

Resolution priority
-------------------
``TURN`` overrides every other axis.  Off-turn samples cross a vertical
regime (``ALT > VZ > GAMMA > UNKVERT``) with a longitudinal regime
(``MACH > CAS > UNK``).

Vertical regime — finite-value membership of ``fdm_alt_sel_ft``,
``fdm_vz_sel_ftmin`` and ``fdm_gamma_sel_rad``.  These columns are
produced by mutually exclusive plateau detectors upstream; if more than
one is finite on a row (defensive case) priority is ``ALT > VZ > GAMMA``.

Longitudinal regime — per-flight constant sub-run detection on
``fdm_mach_sel`` and ``fdm_cas_sel_kt`` (sub-run length >= 2,
``np.ptp < 1e-9``).  When both are constant, ``MACH > CAS``.

Reference
---------
Cui & al., 2019 — "A Trajectory Prediction Method Using Selected
Flight Parameters" — defines the per-mode loss weighting that consumes
``fdm_mode_label`` downstream.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import structlog

__all__ = ["classify_speed_regime", "label_modes"]

_CONST_TOL: float = 1e-9
_MIN_SUBRUN_LEN: int = 2
MODE_LABEL_COLUMN: str = "fdm_mode_label"

_log = structlog.get_logger()


def _runs_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` index pairs for every True run."""
    runs: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _classify_constant(values: np.ndarray) -> np.ndarray:
    """Boolean mask: ``True`` on rows belonging to a constant sub-run >= 2.

    A non-NaN run is split into maximal sub-runs of equal consecutive
    values (``|diff| < 1e-9``).  Sub-runs of length >= 2 are flagged.
    """
    n = len(values)
    constant = np.zeros(n, dtype=bool)
    nonnan = ~np.isnan(values)
    for a, b in _runs_from_mask(nonnan):
        seg = values[a : b + 1]
        change = np.concatenate([[True], np.abs(np.diff(seg)) > _CONST_TOL])
        sub_starts = np.flatnonzero(change)
        sub_ends = np.concatenate([sub_starts[1:], [len(seg)]]) - 1
        for s, e in zip(sub_starts, sub_ends, strict=True):
            if e - s + 1 >= _MIN_SUBRUN_LEN:
                constant[a + s : a + e + 1] = True
    return constant


def classify_speed_regime(df: pl.DataFrame, col: str) -> pl.Series:
    """Boolean mask flagging constant sub-runs in ``col``, per flight.

    Groups by ``meta_flight_id`` (preserving order) and applies
    :func:`_classify_constant` to ``col`` within each flight.  The
    returned series has the same length as ``df`` and aligns with the
    input ordering.

    Args:
        df: Polars DataFrame with a ``meta_flight_id`` column and ``col``.
        col: Name of the float column to scan for constant sub-runs.

    Returns:
        Boolean :class:`pl.Series` (length ``df.height``).  ``True`` where
        the row belongs to a constant sub-run of length >= 2 within its
        flight.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame(
        ...     {
        ...         "meta_flight_id": ["A", "A", "A", "B", "B", "B"],
        ...         "fdm_mach_sel": [0.78, 0.78, 0.78, 0.70, 0.72, 0.74],
        ...     }
        ... )
        >>> classify_speed_regime(df, "fdm_mach_sel").to_list()
        [True, True, True, False, False, False]
    """
    parts: list[np.ndarray] = []
    for (_fid,), fdf in df.group_by("meta_flight_id", maintain_order=True):
        values = fdf[col].to_numpy().astype(np.float64)
        parts.append(_classify_constant(values))
    if not parts:
        return pl.Series(name=f"{col}_const", values=[], dtype=pl.Boolean)
    return pl.Series(name=f"{col}_const", values=np.concatenate(parts), dtype=pl.Boolean)


def _vertical_regime(df: pl.DataFrame) -> np.ndarray:
    """Return per-row vertical regime as a string array.

    Priority ``ALT > VZ > GAMMA > UNKVERT``.  Logs a single warning if
    more than one column is finite on the same row (defensive — the
    upstream plateau detectors are mutually exclusive).
    """
    alt = df["fdm_alt_sel_ft"].to_numpy().astype(np.float64)
    vz = df["fdm_vz_sel_ftmin"].to_numpy().astype(np.float64)
    gamma = df["fdm_gamma_sel_rad"].to_numpy().astype(np.float64)

    alt_f = np.isfinite(alt)
    vz_f = np.isfinite(vz)
    gamma_f = np.isfinite(gamma)

    overlap = (alt_f.astype(int) + vz_f.astype(int) + gamma_f.astype(int)) > 1
    if bool(overlap.any()):
        _log.warning("vert_regime_overlap", n_rows=int(overlap.sum()))

    out = np.full(len(df), "UNKVERT", dtype=object)
    out[gamma_f] = "GAMMA"
    out[vz_f] = "VZ"
    out[alt_f] = "ALT"
    return out


def _longitudinal_regime(df: pl.DataFrame) -> np.ndarray:
    """Return per-row longitudinal regime as a string array.

    Priority ``MACH > CAS > UNK``.
    """
    mach_const = classify_speed_regime(df, "fdm_mach_sel").to_numpy().astype(bool)
    cas_const = classify_speed_regime(df, "fdm_cas_sel_kt").to_numpy().astype(bool)
    out = np.full(len(df), "UNK", dtype=object)
    out[cas_const] = "CAS"
    out[mach_const] = "MACH"
    return out


def label_modes(df: pl.DataFrame) -> pl.DataFrame:
    """Attach the ``fdm_mode_label`` column to ``df``.

    Resolution priority:

    1. ``fdm_in_turn = True`` -> ``"TURN"``.
    2. Otherwise ``"<vertical>_<longitudinal>"`` where the vertical part
       is ``ALT | VZ | GAMMA | UNKVERT`` and the longitudinal part is
       ``MACH | CAS | UNK``.

    Idempotent: an existing ``fdm_mode_label`` column is dropped before
    recomputation, so re-running the function yields the same result
    without producing a ``_2`` suffix.

    Args:
        df: Polars DataFrame with ``meta_flight_id``, ``fdm_in_turn``,
            ``fdm_alt_sel_ft``, ``fdm_vz_sel_ftmin``,
            ``fdm_gamma_sel_rad``, ``fdm_mach_sel`` and
            ``fdm_cas_sel_kt``.

    Returns:
        Same DataFrame with the additional ``fdm_mode_label: pl.Utf8``
        column.  Row count and order are preserved.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame(
        ...     {
        ...         "meta_flight_id": ["A"] * 4,
        ...         "fdm_in_turn": [False] * 4,
        ...         "fdm_alt_sel_ft": [10000.0] * 4,
        ...         "fdm_vz_sel_ftmin": [float("nan")] * 4,
        ...         "fdm_gamma_sel_rad": [float("nan")] * 4,
        ...         "fdm_mach_sel": [0.78] * 4,
        ...         "fdm_cas_sel_kt": [250.0, 252.0, 254.0, 256.0],
        ...     }
        ... )
        >>> label_modes(df)["fdm_mode_label"].to_list()
        ['ALT_MACH', 'ALT_MACH', 'ALT_MACH', 'ALT_MACH']
    """
    if MODE_LABEL_COLUMN in df.columns:
        df = df.drop(MODE_LABEL_COLUMN)

    vert = _vertical_regime(df)
    long = _longitudinal_regime(df)
    in_turn = df["fdm_in_turn"].to_numpy()

    labels = np.array(
        [f"{v}_{lo}" for v, lo in zip(vert, long, strict=True)],
        dtype=object,
    )
    turn_mask = np.where(in_turn == None, False, in_turn.astype(bool))  # noqa: E711
    labels[turn_mask] = "TURN"

    return df.with_columns(pl.Series(MODE_LABEL_COLUMN, labels.tolist(), dtype=pl.Utf8))
