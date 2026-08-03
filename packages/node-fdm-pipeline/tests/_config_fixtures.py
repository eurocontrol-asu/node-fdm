"""Canonical detector tuning shared by every test that builds a pipeline config.

``SelectedParamConfig`` makes its five bilateral channels required (see the note
in ``node_fdm_pipeline.config``), so any config a test loads must declare them.
This module holds that block once, in both YAML and dict form.

These are the values paper_opensky26 retained on its coverage x self-consistency
Pareto front over 1,472 flights — main.tex table 3 — not arbitrary filler. That
matters: a fixture holding invented numbers would be the very default the
required fields exist to abolish, only hidden one level deeper. Tuning that
breaks a test here breaks it against a published calibration, which is a
statement worth reading.

The channels quote sigma_r and slope-tol from the paper. The remaining knobs
(sigma_s, n_passes, flat_tol, min_len, cutoff_s) are the pipeline's own, outside
the swept grid, and are carried over from the values these models shipped with.

It lives beside ``conftest.py`` rather than inside ``src/`` because it is test
support, not shipped code. ``conftest.py`` re-exports everything here, so test
modules can reach it either by importing this module or via the fixtures.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from node_fdm_pipeline.config import (
    AltFilterConfig,
    CasFilterConfig,
    GammaFilterConfig,
    MachFilterConfig,
    SelectedParamConfig,
    VzFilterConfig,
)

__all__ = [
    "SELECTED_PARAMS",
    "SELECTED_PARAMS_YAML",
    "config_yaml",
    "selected_param_config",
    "write_config",
]

SELECTED_PARAMS_YAML = """\
selected_params:
  mach:
    sigma_s: 8.0
    sigma_r: 0.01          # opensky26 tbl.3
    n_passes: 2
    slope_tol: 3.0e-4      # opensky26 tbl.3
    flat_tol: 5.0e-2
    min_len: 15
  cas:
    cutoff_s: 180.0
    sigma_s: 8.0
    sigma_r: 3.0           # opensky26 tbl.3
    n_passes: 2
    slope_tol: 0.09        # opensky26 tbl.3
    flat_tol: 20.0
    min_len: 5
  vz:
    sigma_s: 6.0
    sigma_r: 100.0         # opensky26 tbl.3
    slope_tol: 50.0        # opensky26 tbl.3
    flat_tol: 100.0
    min_len: 10
  alt:
    sigma_s: 6.0
    sigma_r: 20.0          # opensky26 tbl.3
    n_passes: 2
    tol_ftmin: 150.0
    min_len: 6
  gamma:
    sigma_s: 6.0
    sigma_r: 0.002         # opensky26 tbl.3
    slope_tol: 1.2e-3      # opensky26 tbl.3
    flat_tol: 2.0e-3
    abs_min: 5.0e-3
    min_len: 10
"""

#: The same block as nested dicts, for tests building a config in Python.
SELECTED_PARAMS: dict[str, dict[str, float | int]] = {
    "mach": {
        "sigma_s": 8.0,
        "sigma_r": 0.01,
        "n_passes": 2,
        "slope_tol": 3.0e-4,
        "flat_tol": 5.0e-2,
        "min_len": 15,
    },
    "cas": {
        "cutoff_s": 180.0,
        "sigma_s": 8.0,
        "sigma_r": 3.0,
        "n_passes": 2,
        "slope_tol": 0.09,
        "flat_tol": 20.0,
        "min_len": 5,
    },
    "vz": {
        "sigma_s": 6.0,
        "sigma_r": 100.0,
        "slope_tol": 50.0,
        "flat_tol": 100.0,
        "min_len": 10,
    },
    "alt": {
        "sigma_s": 6.0,
        "sigma_r": 20.0,
        "n_passes": 2,
        "tol_ftmin": 150.0,
        "min_len": 6,
    },
    "gamma": {
        "sigma_s": 6.0,
        "sigma_r": 0.002,
        "slope_tol": 1.2e-3,
        "flat_tol": 2.0e-3,
        "abs_min": 5.0e-3,
        "min_len": 10,
    },
}


def selected_param_config() -> SelectedParamConfig:
    """Build a fully-declared :class:`SelectedParamConfig` from the block above.

    Mypy cannot follow ``SelectedParamConfig(**SELECTED_PARAMS)`` — the nested
    dicts are not the per-channel model types it expects — so the construction
    is written out once here instead of being ignored at each call site.
    """
    return SelectedParamConfig(
        mach=MachFilterConfig(**SELECTED_PARAMS["mach"]),  # type: ignore[arg-type]
        cas=CasFilterConfig(**SELECTED_PARAMS["cas"]),  # type: ignore[arg-type]
        vz=VzFilterConfig(**SELECTED_PARAMS["vz"]),  # type: ignore[arg-type]
        alt=AltFilterConfig(**SELECTED_PARAMS["alt"]),  # type: ignore[arg-type]
        gamma=GammaFilterConfig(**SELECTED_PARAMS["gamma"]),  # type: ignore[arg-type]
    )


def config_yaml(
    data_dir: Path | str,
    *,
    typecodes: Sequence[str] | None = ("A320",),
    extra: str = "",
) -> str:
    """Render a complete, valid pipeline config as YAML text.

    Args:
        data_dir: Value for ``paths.data_dir``.
        typecodes: Aircraft typecodes; pass ``[]`` for the empty-list case the
            root validator is expected to reject, or ``None`` to omit the key.
        extra: Additional top-level YAML blocks, appended verbatim.

    Returns:
        YAML text suitable for ``Path.write_text``.
    """
    parts = [f'paths:\n  data_dir: "{data_dir}"\n']
    if typecodes is not None:
        if typecodes:
            listed = "".join(f"  - {code}\n" for code in typecodes)
            parts.append(f"\ntypecodes:\n{listed}")
        else:
            parts.append("\ntypecodes: []\n")
    parts.append("\n" + SELECTED_PARAMS_YAML)
    if extra:
        parts.append("\n" + extra.lstrip("\n"))
    return "".join(parts)


def write_config(
    path: Path,
    data_dir: Path | str,
    *,
    typecodes: Sequence[str] | None = ("A320",),
    extra: str = "",
) -> Path:
    """Write a complete, valid pipeline config YAML to *path*.

    Thin wrapper over :func:`config_yaml` for the common call site.

    Returns:
        *path*, so callers can assign and return in one expression.
    """
    path.write_text(config_yaml(data_dir, typecodes=typecodes, extra=extra))
    return path
