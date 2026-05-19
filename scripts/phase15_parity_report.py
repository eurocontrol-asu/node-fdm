"""Phase 1.5 parity report — CL-mode model vs Newton-mode (L/m_ref) baseline.

Generates the empirical artifacts that gate Phase 1.5 (AXM-1739):

* ``data/models/{cl_model}/parity_report.md`` — loss parity table, b_raw
  drift table, identifiability ratio, verdict (AC2 / AC3 / AC5 / AC6).
* ``data/models/{cl_model}/cl_distribution.md`` — CL percentiles on the
  val-set cruise slice (AC4).

Default model names follow the post-alignment convention (arch ↔ model
numbering matches):

* ``full_hybrid_v3`` — CL-mode model, arch ``node_adsb_hybrid_v3``
  (Phase 1.5, ``L = q · S · (CL_REF + cl_residual)``).
* ``full_hybrid_v2`` — Newton-mode baseline, arch ``node_adsb_hybrid_v2``
  (Phase 1, ``L = (1 + lift_residual_norm) · m·g``, 6 features incl.
  ``mach_cruise_planned``).

Earlier runs ``full_hybrid_v0.legacy`` (pre-MassEncoder, arch
``node_adsb_hybrid_v1``) and ``full_hybrid_v1.legacy`` (5-feature
L/m_ref attempt that did not validate) are archived under
``data/models/`` for historical reference.

Usage (run from repo root once both models are trained)::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/phase15_parity_report.py run \\
        --cl-name full_hybrid_v3 \\
        --newton-name full_hybrid_v2

Subcommands ``parity`` and ``cl-dist`` emit a single artifact each;
``run`` emits both. All artifacts land under the CL-model directory.

**Cruise-slice caveat (AC4)** — the ticket specifies
``fdm_mode_label == "ALT_MACH"`` for the cruise slice. ``fdm_mode_label``
is a delta-table column produced by
``node_fdm_data.preprocessing.label_modes`` but is **not threaded into
``FlightSample`` tensors** by the existing val-set loader. To keep the
"no new fixtures" constraint from technical_notes, this script uses an
altitude proxy ``raw_alt_m > 9144 m`` (~30 000 ft) as the cruise mask.
This slightly over-includes high-altitude non-ALT_MACH segments
(e.g. shallow GAMMA_MACH climbs near cruise alt) but matches the
performance envelope the ticket targets. The fraction included is
logged in ``cl_distribution.md`` so a downstream consumer can re-weight.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cyclopts
import numpy as np
import polars as pl
import structlog
import torch

structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
log = structlog.get_logger("phase15_parity_report")

__all__ = ["app", "main"]

app = cyclopts.App(
    name="phase15_parity_report",
    help="Generate the Phase 1.5 parity + CL-distribution artifacts for AXM-1739.",
)

# ---------------------------------------------------------------------------
# Thresholds (kept here so a re-run with tightened criteria stays auditable)
# ---------------------------------------------------------------------------

PARITY_RATIO_TOL: float = 0.02
"""AC3: |val_loss_cl - val_loss_newton| / val_loss_newton <= 0.02 to PASS."""

IDENTIFIABILITY_THRESHOLD: float = 1.20
"""AC5: identifiability ratio strictly > 1.20 to PASS."""

B_RAW_DRIFT_TOL: float = 0.10
"""AC6: |coef_cl - coef_newton| / |coef_newton| <= 0.10 per feature to PASS."""

CL_CRUISE_MEDIAN_RANGE: tuple[float, float] = (0.4, 0.6)
"""AC4(a): median(CL[cruise]) must fall inside this band."""

CL_CRUISE_P10_P90_RANGE: tuple[float, float] = (0.3, 0.7)
"""AC4(b): [p10, p90] of CL[cruise] must be contained in this band."""

CRUISE_ALT_MIN_M: float = 9144.0
"""~30 000 ft. Altitude-only proxy for the ALT_MACH cruise slice (see module docstring)."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossSummary:
    """Final-epoch + best-epoch loss snapshot for one model."""

    name: str
    epochs: list[float]
    train_losses: list[float]
    val_losses: list[float]

    @property
    def final_train(self) -> float:
        return self.train_losses[-1] if self.train_losses else float("nan")

    @property
    def final_val(self) -> float:
        return self.val_losses[-1] if self.val_losses else float("nan")

    @property
    def best_val(self) -> float:
        return min(self.val_losses) if self.val_losses else float("nan")

    @property
    def best_epoch(self) -> float:
        if not self.val_losses:
            return float("nan")
        idx = int(np.argmin(self.val_losses))
        return self.epochs[idx]


# ---------------------------------------------------------------------------
# CSV / meta loading
# ---------------------------------------------------------------------------


def _load_loss_summary(model_dir: Path, name: str) -> LossSummary:
    """Parse ``training_losses.csv`` for a model directory."""
    csv_path = model_dir / "training_losses.csv"
    if not csv_path.exists():
        msg = f"training_losses.csv not found under {model_dir}"
        raise FileNotFoundError(msg)
    epochs: list[float] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            epochs.append(float(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
    return LossSummary(name=name, epochs=epochs, train_losses=train_losses, val_losses=val_losses)


def _feature_cols_from_state(state: dict[str, torch.Tensor]) -> list[str]:
    """Recover feature_cols ordering from a MassEncoder state dict.

    The normalizer registers ``mean_<col>`` / ``std_<col>`` buffers for
    each feature column at construction time, so the trained checkpoint
    carries the authoritative column list — even when the architecture
    spec on disk has drifted (e.g. a checkpoint trained with
    FLIGHT_FEATURE_COLS_6 saved under an arch that now resolves to 5
    features via the registry).
    """
    prefix = "normalizer.mean_"
    return [key[len(prefix) :] for key in state if key.startswith(prefix)]


def _load_mass_coefficients(model_dir: Path) -> dict[str, float]:
    """Load mass_encoder.pt and reconstruct effective_coefficients() output.

    Mirrors ``MassEncoderLinear.effective_coefficients`` (post-softplus,
    signed). Feature columns are recovered from the checkpoint's
    normalizer buffers rather than the architecture spec — this keeps the
    drift report robust against transitional states where a trained
    baseline was built with a different ``flight_feature_cols`` than the
    currently-registered arch spec exposes.
    """
    pt_path = model_dir / "mass_encoder.pt"
    if not pt_path.exists():
        msg = f"mass_encoder.pt not found under {model_dir}"
        raise FileNotFoundError(msg)
    state = torch.load(pt_path, weights_only=True, map_location="cpu")
    b_raw = state["b_raw"]
    b0 = state["b0"]
    signs = state["signs"]
    b_effective = signs * torch.nn.functional.softplus(b_raw)
    feature_cols = _feature_cols_from_state(state)
    if len(feature_cols) != b_raw.shape[0]:
        msg = (
            f"feature_cols recovered from state ({len(feature_cols)}) does not match "
            f"b_raw shape ({b_raw.shape[0]}) under {model_dir}"
        )
        raise RuntimeError(msg)
    coefficients: dict[str, float] = {"b0": float(b0.item())}
    for col, value in zip(feature_cols, b_effective, strict=True):
        coefficients[col] = float(value.item())
    return coefficients


# ---------------------------------------------------------------------------
# Trainer construction (mirrors scripts/analyze_mass_encoder.py)
# ---------------------------------------------------------------------------


def _build_trainer_for_model(
    model_name: str, config_path: Path, device: str, val_limit: int | None
):
    """Reload a trained model in-process. Returns ``(trainer, val_dataset, spec, meta)``.

    Heavy imports are local so the script's ``--help`` output stays fast and
    test-suite collection doesn't require torch.
    """
    # Architecture modules self-register on import — kept as bare imports
    # to make the registry contract explicit at the script entry point.
    import node_fdm.architectures.adsb_hybrid
    import node_fdm.architectures.adsb_hybrid_v2
    import node_fdm.architectures.adsb_hybrid_v3  # noqa: F401
    from node_fdm.architectures.registry import get as get_arch_spec
    from node_fdm.loader import get_train_val_data
    from node_fdm.trainer import ODETrainer, TrainingConfig
    from node_fdm_data.delta import read_delta_table
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config_path)
    models_dir = cfg.paths.resolve("models_dir")
    model_dir = models_dir / model_name
    if not model_dir.exists():
        msg = f"model directory missing: {model_dir}"
        raise FileNotFoundError(msg)

    meta = json.loads((model_dir / "meta.json").read_text())
    spec = get_arch_spec(meta["architecture_name"])

    delta_path = cfg.paths.resolve("delta_table")
    df = read_delta_table(delta_path).filter(pl.col("fdm_flag_valid"))
    if cfg.typecodes:
        df = df.filter(pl.col("meta_aircraft_type").is_in(cfg.typecodes))

    dx_col_names = [c for _, c in spec.dx_cols]
    flight_feature_cols = list(spec.flight_feature_cols or [])
    # We only need val_ds for identifiability_test / CL forward pass, but
    # get_train_val_data refuses an empty train set, and small flight
    # caps can produce 0 windows after the `require_routing` filter
    # cascades (see flights_dropped_no_routing logs). Load the full
    # train set (~5s on the matrix) to stay safe — train is never used
    # downstream so this only costs memory.
    train_ds, val_ds = get_train_val_data(
        data_df=df,
        x_cols=spec.x_cols,
        u_cols=spec.u_cols,
        e_cols=spec.e0_cols,
        e1_cols=spec.e1_cols,
        dx_cols=dx_col_names,
        seq_len=meta["seq_len"],
        shift=meta["shift"],
        train_limit=None,
        val_limit=val_limit,
        flight_feature_cols=flight_feature_cols or None,
        require_routing=True,
    )

    config = TrainingConfig(
        architecture_name=meta["architecture_name"],
        model_name=model_name,
        model_params=tuple(meta["model_params"]),
        step=meta["step"],
        shift=meta["shift"],
        seq_len=meta["seq_len"],
        lr=meta["lr"],
        batch_size=meta["batch_size"],
        epochs=meta["epochs"],
        method=meta["method"],
        seed=meta.get("seed"),
        activation=meta.get("activation", "silu"),
        use_mode_weights=meta.get("use_mode_weights", False),
        mode_weight_alpha=meta.get("mode_weight_alpha", 0.5),
    )
    trainer = ODETrainer(
        config=config,
        train_dataset=train_ds,  # tiny placeholder; identifiability_test only uses val.
        val_dataset=val_ds,
        model_dir=models_dir,
        device=device,
    )
    trainer.load_model_weights()
    return trainer, val_ds, spec, meta


# ---------------------------------------------------------------------------
# CL distribution (AC4)
# ---------------------------------------------------------------------------


#: Altitude bands (m) for the phase-stratified diagnostic. Cruise band
#: uses CRUISE_ALT_MIN_M. Low + mid cover the climb/descent regimes
#: where CL deviates most strongly from CL_REF=0.5.
ALT_BAND_LOW_MAX_M: float = 3048.0  # 10 000 ft
ALT_BAND_MID_MAX_M: float = 9144.0  # 30 000 ft (matches CRUISE_ALT_MIN_M)


@dataclass(frozen=True)
class PhaseStats:
    """Target cl_residual statistics in a single altitude band."""

    n_steps: int
    target_mean: float
    target_std: float
    target_p10: float
    target_p90: float
    cl_steady_mean: float
    """Mean of ``m_ref·g / (q·S_REF)`` -- the CL that exactly balances
    weight at steady-state in this altitude band. The gap between this
    and ``CL_REF=0.5`` is what the NN must learn as ``cl_residual``."""


@dataclass(frozen=True)
class ClDiagnostics:
    """Predicted vs target CL/residual distributions over the val set.

    Captures four populations:

    * ``pred_cruise`` / ``target_cruise`` — CL on the cruise slice
      (``raw_alt_m > CRUISE_ALT_MIN_M``). ``pred`` comes from the NN
      forward (``CL_REF + cl_residual_pred``); ``target`` comes from the
      analytical inverse of PhysicsLayer (``_compute_cl_residual``).
    * ``pred_all`` / ``target_all`` — same two quantities, on every val
      step (no cruise filter). Used to compare the operational regime
      vs the full training-data distribution that drives the normalizer.

    Plus a phase-stratified breakdown (low / mid / cruise altitude
    bands) of the **target** ``cl_residual`` and the ``CL_steady`` that
    would balance weight at each altitude. Diagnostic question: does
    the target residual amplitude scale strongly with phase (because
    CL_REF is a constant while the operational CL is phase-dependent)?
    """

    pred_all: np.ndarray
    pred_cruise: np.ndarray
    target_all: np.ndarray
    target_cruise: np.ndarray
    # Raw residuals (CL - CL_steady_q) on all val + on the cruise slice.
    # Used for the §5 normalizer drift diagnostic: the magnitude of
    # ``|residual|`` is what the normalizer ``p999`` is meant to bound.
    target_residual_all: np.ndarray
    target_residual_cruise: np.ndarray
    n_total: int
    n_cruise: int
    phase_low: PhaseStats
    phase_mid: PhaseStats
    phase_cruise: PhaseStats


def _phase_stats(
    target_residual: np.ndarray,
    q_pa: np.ndarray,
    mask: np.ndarray,
    *,
    m_ref_kg: float,
    s_ref_m2: float,
    g: float,
) -> PhaseStats:
    """Compute per-band stats. Empty mask returns NaNs."""
    n = int(mask.sum())
    if n == 0:
        return PhaseStats(
            n_steps=0,
            target_mean=float("nan"),
            target_std=float("nan"),
            target_p10=float("nan"),
            target_p90=float("nan"),
            cl_steady_mean=float("nan"),
        )
    sel_res = target_residual[mask]
    sel_q = q_pa[mask]
    # CL_steady = m_ref·g / (q·S) -- the CL that exactly balances weight
    # at this altitude/speed combination. Filter pathological q (near-zero
    # at very low TAS during initial climb) by clipping at 100 Pa.
    sel_q_safe = np.maximum(sel_q, 100.0)
    cl_steady = m_ref_kg * g / (sel_q_safe * s_ref_m2)
    return PhaseStats(
        n_steps=n,
        target_mean=float(np.mean(sel_res)),
        target_std=float(np.std(sel_res)),
        target_p10=float(np.percentile(sel_res, 10)),
        target_p90=float(np.percentile(sel_res, 90)),
        cl_steady_mean=float(np.mean(cl_steady)),
    )


def _collect_cl_diagnostics(trainer, val_dataset, spec) -> ClDiagnostics:
    """Forward-pass the val set through trajectory + data_ode_long.

    Also computes the analytical target ``cl_residual`` (inverse
    PhysicsLayer) on the same samples so predicted and target
    distributions can be compared directly, and stratifies the target
    by altitude band to test the shape hypothesis (CL_REF=0.5 too far
    from the operating CL outside cruise).

    ``data_ode_long`` consumes inputs from x / u / e0 plus e1 columns
    (mach, q_pa, d_alt_ms, ...) that are computed on the fly by
    ``TrajectoryLayer``. So we run ``trajectory`` first, merge its
    outputs into the input dict, then call ``data_ode_long``.
    """
    from node_fdm.dataset import _M_REF_KG, _compute_cl_residual
    from node_fdm.layers.physics import S_REF_A320_M2, cl_baseline_np
    from node_fdm_data.physics.constants import G

    traj_layer = trainer.model.layers_dict["trajectory"]
    long_layer = trainer.model.layers_dict["data_ode_long"]
    traj_layer.eval()
    long_layer.eval()

    alt_idx = spec.x_cols.index("raw_alt_m")
    x_col_index = {name: idx for idx, name in enumerate(spec.x_cols)}
    u_col_index = {name: idx for idx, name in enumerate(spec.u_cols)}
    e_col_index = {name: idx for idx, name in enumerate(spec.e0_cols)}
    e1_col_index = {name: idx for idx, name in enumerate(spec.e1_cols)}
    dx_col_index = {name: idx for idx, (_, name) in enumerate(spec.dx_cols)}
    e0_col_names = list(spec.e0_cols)
    dx_col_names = [name for _, name in spec.dx_cols]

    pred_all: list[float] = []
    pred_cruise: list[float] = []
    target_all: list[float] = []
    target_cruise: list[float] = []
    # Accumulate per-step residual + q + alt across the val set for the
    # phase breakdown done at the end (avoids three passes).
    all_residual: list[float] = []
    all_q: list[float] = []
    all_alt: list[float] = []
    n_total = 0
    n_cruise = 0
    with torch.no_grad():
        for sample in val_dataset:
            inputs: dict[str, torch.Tensor] = {}
            for name, idx in x_col_index.items():
                inputs[name] = sample.x[..., idx]
            for name, idx in u_col_index.items():
                inputs[name] = sample.u[..., idx]
            for name, idx in e_col_index.items():
                inputs[name] = sample.e[..., idx]
            if sample.e1 is not None:
                n_e1 = sample.e1.shape[-1]
                for name, idx in e1_col_index.items():
                    if idx < n_e1:
                        inputs[name] = sample.e1[..., idx]
            for name, idx in dx_col_index.items():
                inputs[name] = sample.dx[..., idx]

            # Predicted CL = CL_REF + NN(cl_residual)
            traj_out = traj_layer(inputs)
            enriched = {**inputs, **traj_out}
            output = long_layer(enriched)
            if "fdm_cl_residual" not in output:
                msg = (
                    "data_ode_long output missing 'fdm_cl_residual' — "
                    "expected the CL-mode arch (node_adsb_hybrid_v3)"
                )
                raise RuntimeError(msg)
            cl_residual_pred = output["fdm_cl_residual"].detach().cpu().numpy().reshape(-1)

            # Target cl_residual from the analytical inverse PhysicsLayer.
            x_arr = sample.x.detach().cpu().numpy().reshape(-1, len(spec.x_cols))
            e_arr = sample.e.detach().cpu().numpy().reshape(-1, len(spec.e0_cols))
            dx_arr = sample.dx.detach().cpu().numpy().reshape(-1, len(spec.dx_cols))
            cl_residual_target = _compute_cl_residual(
                x_arr, e_arr, dx_arr, list(spec.x_cols), e0_col_names, dx_col_names
            )

            alt = sample.x[..., alt_idx].detach().cpu().numpy().reshape(-1)
            cruise_mask = alt > CRUISE_ALT_MIN_M
            n_total += int(alt.size)
            n_cruise += int(cruise_mask.sum())

            # q from TrajectoryLayer output (recomputed from alt/V using
            # the same ISA pressure + ERA5 temperature pipeline as the
            # PhysicsLayer feeds at training time).
            q_pa = traj_out["fdm_q_pa"].detach().cpu().numpy().reshape(-1)
            # Use m_ref for the stats-side baseline (consistent with
            # _compute_cl_residual). At runtime PhysicsLayer uses the
            # true state mass, but the val forward we run here through
            # data_ode_long directly does not propagate mass via
            # MassEncoder -- fdm_mass_kg in sample.x is zero-padded.
            m_ref_arr = np.full_like(q_pa, _M_REF_KG, dtype=np.float64)
            cl_steady = cl_baseline_np(q_pa, m_ref_arr)

            # CL_physical = CL_steady(q) + residual — symmetric to the
            # PhysicsLayer forward + dataset inverse.
            pred_cl = cl_residual_pred + cl_steady
            target_cl = cl_residual_target + cl_steady
            pred_all.extend(pred_cl.tolist())
            target_all.extend(target_cl.tolist())
            pred_cruise.extend(pred_cl[cruise_mask].tolist())
            target_cruise.extend(target_cl[cruise_mask].tolist())

            all_residual.extend(cl_residual_target.tolist())
            all_q.extend(q_pa.tolist())
            all_alt.extend(alt.tolist())

    residual_arr = np.asarray(all_residual, dtype=np.float64)
    q_arr = np.asarray(all_q, dtype=np.float64)
    alt_arr = np.asarray(all_alt, dtype=np.float64)

    low_mask = alt_arr <= ALT_BAND_LOW_MAX_M
    mid_mask = (alt_arr > ALT_BAND_LOW_MAX_M) & (alt_arr <= ALT_BAND_MID_MAX_M)
    cruise_band_mask = alt_arr > ALT_BAND_MID_MAX_M

    stats_kwargs = {"m_ref_kg": _M_REF_KG, "s_ref_m2": S_REF_A320_M2, "g": G}
    phase_low = _phase_stats(residual_arr, q_arr, low_mask, **stats_kwargs)
    phase_mid = _phase_stats(residual_arr, q_arr, mid_mask, **stats_kwargs)
    phase_cruise = _phase_stats(residual_arr, q_arr, cruise_band_mask, **stats_kwargs)

    return ClDiagnostics(
        pred_all=np.asarray(pred_all, dtype=np.float64),
        pred_cruise=np.asarray(pred_cruise, dtype=np.float64),
        target_all=np.asarray(target_all, dtype=np.float64),
        target_cruise=np.asarray(target_cruise, dtype=np.float64),
        target_residual_all=residual_arr,
        target_residual_cruise=residual_arr[cruise_band_mask],
        n_total=n_total,
        n_cruise=n_cruise,
        phase_low=phase_low,
        phase_mid=phase_mid,
        phase_cruise=phase_cruise,
    )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _format_loss_overlay(cl: LossSummary, newton: LossSummary) -> str:
    """Build a Markdown table aligning per-epoch losses for both models."""
    n = max(len(cl.epochs), len(newton.epochs))
    rows = [
        "| epoch | train_cl | val_cl | train_newton | val_newton |",
        "|------:|---------:|-------:|-------------:|-----------:|",
    ]
    for i in range(n):
        epoch = cl.epochs[i] if i < len(cl.epochs) else newton.epochs[i]
        train_cl = f"{cl.train_losses[i]:.6f}" if i < len(cl.train_losses) else "—"
        val_cl = f"{cl.val_losses[i]:.6f}" if i < len(cl.val_losses) else "—"
        train_newton = (
            f"{newton.train_losses[i]:.6f}" if i < len(newton.train_losses) else "—"
        )
        val_newton = f"{newton.val_losses[i]:.6f}" if i < len(newton.val_losses) else "—"
        rows.append(
            f"| {epoch:g} | {train_cl} | {val_cl} | {train_newton} | {val_newton} |"
        )
    return "\n".join(rows)


def _format_b_raw_drift(
    cl_coefs: dict[str, float], newton_coefs: dict[str, float]
) -> tuple[str, bool]:
    """Compare per-feature post-softplus coefficients. Returns (markdown, passed)."""
    common_keys = [k for k in newton_coefs if k in cl_coefs]
    rows = [
        "| feature | newton | cl | |Δ|/|newton| | sign-consistent | within 10% |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    all_pass = True
    for key in common_keys:
        nv = newton_coefs[key]
        cv = cl_coefs[key]
        denom = abs(nv) if abs(nv) > 1e-9 else float("inf")
        drift = abs(cv - nv) / denom
        sign_ok = (nv >= 0) == (cv >= 0)
        within = drift <= B_RAW_DRIFT_TOL
        passed = within or sign_ok
        if not passed:
            all_pass = False
        rows.append(
            f"| `{key}` | {nv:+.4f} | {cv:+.4f} | {drift:.3f} | "
            f"{'✅' if sign_ok else '❌'} | {'✅' if within else '❌'} |"
        )
    return "\n".join(rows), all_pass


def _write_parity_report(
    out_path: Path,
    cl: LossSummary,
    newton: LossSummary,
    cl_coefs: dict[str, float],
    newton_coefs: dict[str, float],
    identifiability: dict[str, float],
    cl_meta: dict,
    newton_meta: dict,
) -> dict[str, object]:
    """Emit ``parity_report.md`` covering AC2, AC3, AC5, AC6. Returns verdict dict."""
    val_loss_cl = cl.final_val
    val_loss_newton = newton.final_val
    if val_loss_newton > 0:
        ratio = (val_loss_cl - val_loss_newton) / val_loss_newton
    else:
        ratio = float("inf")
    parity_pass = abs(ratio) <= PARITY_RATIO_TOL
    drift_md, drift_pass = _format_b_raw_drift(cl_coefs, newton_coefs)
    ident_pass = identifiability["ratio"] > IDENTIFIABILITY_THRESHOLD

    if not parity_pass:
        status = "regression"
    elif not (drift_pass and ident_pass):
        status = "parity (with caveats)"
    else:
        status = "parity"

    overlay_md = _format_loss_overlay(cl, newton)

    lines = [
        f"# Phase 1.5 parity report — {cl.name} (CL-mode) vs {newton.name} (Newton-mode)",
        "",
        f"> **Status**: `{status}`",
        "",
        "This report is emitted by `scripts/phase15_parity_report.py` and covers",
        "AC2 / AC3 / AC5 / AC6 of AXM-1739.",
        "",
        "## 1. Setup",
        "",
        "| Field | CL-mode (Phase 1.5) | Newton-mode (Phase 1 baseline) |",
        "|---|---|---|",
        f"| Model name | `{cl.name}` | `{newton.name}` |",
        f"| Architecture | `{cl_meta['architecture_name']}` "
        f"| `{newton_meta['architecture_name']}` |",
        f"| Epochs (configured) | {cl_meta['epochs']} | {newton_meta['epochs']} |",
        f"| Seed | {cl_meta.get('seed')} | {newton_meta.get('seed')} |",
        f"| Batch | {cl_meta['batch_size']} | {newton_meta['batch_size']} |",
        f"| Seq len | {cl_meta['seq_len']} | {newton_meta['seq_len']} |",
        f"| lr | {cl_meta['lr']} | {newton_meta['lr']} |",
        "",
        "## 2. Loss summary (AC2)",
        "",
        "| Metric | CL | Newton | (CL - Newton) / Newton |",
        "|---|---:|---:|---:|",
        f"| final train_loss | {cl.final_train:.6f} | {newton.final_train:.6f} | — |",
        f"| final val_loss   | {cl.final_val:.6f} | {newton.final_val:.6f} | {ratio:+.4f} |",
        f"| best  val_loss   | {cl.best_val:.6f} | {newton.best_val:.6f} | — |",
        f"| best  epoch      | {cl.best_epoch:g} | {newton.best_epoch:g} | — |",
        "",
        "## 3. Parity criterion (AC3)",
        "",
        f"AC3 requires `|val_loss_cl - val_loss_newton| / val_loss_newton <= {PARITY_RATIO_TOL}`.",
        f"Observed: `{abs(ratio):.4f}` → **{'PASS' if parity_pass else 'FAIL'}**.",
        "",
    ]
    if not parity_pass:
        lines.extend(
            [
                "### Diagnostic (parity fail)",
                "",
                "Walk the technical_notes checklist when investigating:",
                "",
                "1. `cl_residual` head init bias and scale — should default to ~0 since",
                "   `CL_REF=0.5` carries the cruise center. A non-zero init bias on",
                "   `fdm_cl_residual` shifts the operating point of the lift residual.",
                "2. `fdm_q_pa` threading through the `StructuredLayer ⇄ PhysicsLayer`",
                "   interface — confirm the per-step `q` matches dataset stats",
                "   (`p999 ≈ 17 700 Pa` on the current matrix).",
                "3. Normalizer p999 for `fdm_cl_residual` (DERIVED_FEATURES entry from",
                "   AXM-1738) — a too-loose p999 starves the head of gradient signal.",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Per-epoch overlay (AC2)",
            "",
            overlay_md,
            "",
            "## 5. Identifiability test (AC5)",
            "",
            f"`trainer.identifiability_test(factor=1.3)` on `{cl.name}` (CL-mode):",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| baseline_mse  | {identifiability['baseline_mse']:.6f} |",
            f"| perturbed_mse | {identifiability['perturbed_mse']:.6f} |",
            f"| ratio         | {identifiability['ratio']:.4f} |",
            "",
            f"AC5 requires `ratio > {IDENTIFIABILITY_THRESHOLD}` → "
            f"**{'PASS' if ident_pass else 'FAIL'}**.",
            "",
            "## 6. MassEncoder b_raw drift (AC6)",
            "",
            "Effective (post-softplus, signed) coefficients. AC6 passes per-feature if",
            f"either `|Δ|/|newton| ≤ {B_RAW_DRIFT_TOL}` or the sign is preserved.",
            "",
            drift_md,
            "",
            f"Overall AC6 verdict: **{'PASS' if drift_pass else 'FAIL'}**.",
            "",
            "## 7. Verdict",
            "",
            f"`status: {status}`",
            "",
            "- AC2 (overlay generated): ✅",
            f"- AC3 (parity <= {PARITY_RATIO_TOL}): {'✅' if parity_pass else '❌'}",
            f"- AC5 (identifiability > {IDENTIFIABILITY_THRESHOLD}): "
            f"{'✅' if ident_pass else '❌'}",
            f"- AC6 (b_raw drift <= {B_RAW_DRIFT_TOL} or sign-consistent): "
            f"{'✅' if drift_pass else '❌'}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))
    return {
        "status": status,
        "parity_ratio": ratio,
        "parity_pass": parity_pass,
        "identifiability_ratio": identifiability["ratio"],
        "identifiability_pass": ident_pass,
        "drift_pass": drift_pass,
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    """Return median, p10, p90, p999, std, mean on a flat distribution."""
    if values.size == 0:
        return {k: float("nan") for k in ("median", "p10", "p90", "p999", "std", "mean")}
    finite = values[np.isfinite(values)]
    return {
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "p999": float(np.percentile(finite, 99.9)),
        "std": float(np.std(finite)),
        "mean": float(np.mean(finite)),
    }


def _write_cl_distribution(
    out_path: Path,
    diag: ClDiagnostics,
    model_name: str,
    normalizer_stats: dict[str, float] | None,
) -> dict[str, object]:
    """Emit ``cl_distribution.md`` covering AC4 + normalizer drift diagnostic.

    Compares predicted vs analytical-target CL on every val step and on
    the cruise slice, and contrasts the cruise operational ``p999`` of
    the target residual against the global normalizer ``p999`` saved in
    ``meta.json`` (when available). A large gap is a smoking gun for the
    "p999 trop large" hypothesis raised in the parity diagnostic.
    """
    if diag.pred_cruise.size == 0:
        out_path.write_text(
            "# CL distribution\n\n"
            f"No cruise steps found in val set for `{model_name}` "
            f"(altitude proxy > {CRUISE_ALT_MIN_M:g} m).\n"
        )
        return {
            "median_cl_cruise": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "median_in_band": False,
            "p10_p90_in_band": False,
            "p999_ratio_global_over_cruise": float("nan"),
        }

    pc = _stats(diag.pred_cruise)
    pa = _stats(diag.pred_all)
    tc = _stats(diag.target_cruise)
    ta = _stats(diag.target_all)

    median_in_band = CL_CRUISE_MEDIAN_RANGE[0] <= pc["median"] <= CL_CRUISE_MEDIAN_RANGE[1]
    p10_p90_in_band = (
        CL_CRUISE_P10_P90_RANGE[0] <= pc["p10"] <= CL_CRUISE_P10_P90_RANGE[1]
        and CL_CRUISE_P10_P90_RANGE[0] <= pc["p90"] <= CL_CRUISE_P10_P90_RANGE[1]
    )

    # The residual distribution is what the normalizer p999 is sized to
    # bound. Compute p999 of |residual| directly on the raw residual
    # arrays (target_residual_cruise / target_residual_all) — these are
    # the analytical inverse output, already centered on 0 by the
    # q-dependent baseline CL_steady(q).
    from node_fdm.layers.physics import CL_REF

    tc_res_p999 = (
        float(np.percentile(np.abs(diag.target_residual_cruise), 99.9))
        if diag.target_residual_cruise.size
        else float("nan")
    )
    ta_res_p999 = (
        float(np.percentile(np.abs(diag.target_residual_all), 99.9))
        if diag.target_residual_all.size
        else float("nan")
    )
    normalizer_p999 = (normalizer_stats or {}).get("p999", float("nan"))

    if tc_res_p999 and np.isfinite(tc_res_p999) and tc_res_p999 > 0:
        ratio_norm = normalizer_p999 / tc_res_p999
        ratio_all = ta_res_p999 / tc_res_p999
    else:
        ratio_norm = float("nan")
        ratio_all = float("nan")

    cruise_fraction = diag.n_cruise / max(diag.n_total, 1)

    # Helper for §3 row format — keeps body lines short (E501)
    def _delta(p: float, t: float) -> str:
        return f"{p - t:+.4f}"

    lines = [
        f"# CL distribution — `{model_name}`",
        "",
        "AC4 of AXM-1739: validate the CL operating range of the CL-mode model on",
        f"the val-set cruise slice. Cruise mask = `raw_alt_m > {CRUISE_ALT_MIN_M:g} m`",
        "(altitude proxy for `fdm_mode_label == ALT_MACH`; see module docstring).",
        "",
        "## 1. Slice coverage",
        "",
        f"- Total val steps: `{diag.n_total}`",
        f"- Cruise-slice steps: `{diag.n_cruise}` ({100 * cruise_fraction:.1f}%)",
        "",
        "## 2. CL = CL_REF + cl_residual_pred — predicted percentiles (AC4)",
        "",
        "| Metric | Value | Target | PASS |",
        "|---|---:|---|:---:|",
        f"| `median_CL_cruise` | {pc['median']:.4f} | "
        f"[{CL_CRUISE_MEDIAN_RANGE[0]}, {CL_CRUISE_MEDIAN_RANGE[1]}] | "
        f"{'✅' if median_in_band else '❌'} |",
        f"| `p10` | {pc['p10']:.4f} | \\geq {CL_CRUISE_P10_P90_RANGE[0]} | "
        f"{'✅' if pc['p10'] >= CL_CRUISE_P10_P90_RANGE[0] else '❌'} |",
        f"| `p90` | {pc['p90']:.4f} | \\leq {CL_CRUISE_P10_P90_RANGE[1]} | "
        f"{'✅' if pc['p90'] <= CL_CRUISE_P10_P90_RANGE[1] else '❌'} |",
        "",
        f"AC4 (a + b) verdict: **{'PASS' if median_in_band and p10_p90_in_band else 'FAIL'}**.",
        "",
        "## 3. Predicted vs target CL — cruise slice",
        "",
        "`target` is the analytical inverse of `PhysicsLayer` "
        "(`_compute_cl_residual` in `dataset.py`) -- the value the NN head "
        "should learn to emit, given (gamma, V, q, d_gamma, m_ref). `pred` "
        "is what the trained head actually outputs.",
        "",
        "| Statistic | predicted | target | pred - target |",
        "|---|---:|---:|---:|",
        f"| mean | {pc['mean']:+.4f} | {tc['mean']:+.4f} | {_delta(pc['mean'], tc['mean'])} |",
        f"| std  | {pc['std']:.4f} | {tc['std']:.4f} | {_delta(pc['std'], tc['std'])} |",
        f"| p10  | {pc['p10']:.4f} | {tc['p10']:.4f} | {_delta(pc['p10'], tc['p10'])} |",
        f"| median | {pc['median']:.4f} | {tc['median']:.4f} | "
        f"{_delta(pc['median'], tc['median'])} |",
        f"| p90  | {pc['p90']:.4f} | {tc['p90']:.4f} | {_delta(pc['p90'], tc['p90'])} |",
        "",
        "## 4. Pred vs target CL — all val (cruise + non-cruise)",
        "",
        "| Statistic | predicted | target | pred - target |",
        "|---|---:|---:|---:|",
        f"| mean | {pa['mean']:+.4f} | {ta['mean']:+.4f} | {_delta(pa['mean'], ta['mean'])} |",
        f"| std  | {pa['std']:.4f} | {ta['std']:.4f} | {_delta(pa['std'], ta['std'])} |",
        f"| p10  | {pa['p10']:.4f} | {ta['p10']:.4f} | {_delta(pa['p10'], ta['p10'])} |",
        f"| median | {pa['median']:.4f} | {ta['median']:.4f} | "
        f"{_delta(pa['median'], ta['median'])} |",
        f"| p90  | {pa['p90']:.4f} | {ta['p90']:.4f} | {_delta(pa['p90'], ta['p90'])} |",
        "",
        "## 5. Cruise vs global — target residual p999 (diagnostic for AC3 fail)",
        "",
        "The CL normalizer's `p999` is set from the training-data global "
        "distribution (all phases, including high-bank manoeuvres and "
        "low-altitude transients). The training **objective** lives in "
        "cruise. If the global `p999` is much larger than the cruise "
        "operational `p999`, the NN head output cap is sized for "
        "extreme regimes and the cruise signal gets compressed close to "
        "zero in normalized space, starving gradient.",
        "",
        "| Quantity | value |",
        "|---|---:|",
        f"| `|cl_residual_target|.p999` on cruise | {tc_res_p999:.4f} |",
        f"| `|cl_residual_target|.p999` on all val | {ta_res_p999:.4f} |",
        f"| normalizer `p999` (meta.json `stats_dict['fdm_cl_residual']`) | "
        f"{normalizer_p999:.4f} |",
        f"| ratio normalizer / cruise | {ratio_norm:.2f}x |",
        f"| ratio all-val / cruise | {ratio_all:.2f}x |",
        "",
        "**Interpretation** -- a large `normalizer / cruise` ratio means "
        "the residual amplitude varies a lot **across phases**. Two "
        "competing readings: (a) the normalizer is sized for noise that "
        "the NN should learn to ignore; (b) the NN must legitimately "
        "represent very different CL values per phase because `CL_REF=0.5` "
        "is far from steady-state CL outside cruise. See S6 (phase "
        "breakdown) to disambiguate.",
        "",
        "## 6. Target residual amplitude by altitude band (shape diagnostic)",
        "",
        "If the model is genuinely multi-phase, `cl_residual` must shift "
        "to compensate for the gap between `CL_REF=0.5` and `CL_steady = "
        "m_ref·g / (q·S_REF)` -- the CL that exactly balances weight "
        "given the current dynamic pressure. The wider that gap, the "
        "harder the NN has to work; the more phase-dependent the "
        "amplitude, the worse a constant `CL_REF` performs.",
        "",
        "Bands: `low = alt <= 3048 m (10 kft)`, "
        "`mid = 3048 < alt <= 9144 m`, `cruise = alt > 9144 m`.",
        "",
        "| band | n_steps | target_mean | target_std | target_p10 | "
        "target_p90 | CL_steady_mean | |CL_steady - CL_REF| |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| low | {diag.phase_low.n_steps} | "
        f"{diag.phase_low.target_mean:+.4f} | "
        f"{diag.phase_low.target_std:.4f} | "
        f"{diag.phase_low.target_p10:+.4f} | "
        f"{diag.phase_low.target_p90:+.4f} | "
        f"{diag.phase_low.cl_steady_mean:.4f} | "
        f"{abs(diag.phase_low.cl_steady_mean - CL_REF):.4f} |",
        f"| mid | {diag.phase_mid.n_steps} | "
        f"{diag.phase_mid.target_mean:+.4f} | "
        f"{diag.phase_mid.target_std:.4f} | "
        f"{diag.phase_mid.target_p10:+.4f} | "
        f"{diag.phase_mid.target_p90:+.4f} | "
        f"{diag.phase_mid.cl_steady_mean:.4f} | "
        f"{abs(diag.phase_mid.cl_steady_mean - CL_REF):.4f} |",
        f"| cruise | {diag.phase_cruise.n_steps} | "
        f"{diag.phase_cruise.target_mean:+.4f} | "
        f"{diag.phase_cruise.target_std:.4f} | "
        f"{diag.phase_cruise.target_p10:+.4f} | "
        f"{diag.phase_cruise.target_p90:+.4f} | "
        f"{diag.phase_cruise.cl_steady_mean:.4f} | "
        f"{abs(diag.phase_cruise.cl_steady_mean - CL_REF):.4f} |",
        "",
        "**Reading the table**:",
        "",
        "* If `target_std` is roughly constant across bands -> shape OK, "
        "the multi-phase signal is mostly noise around a phase-dependent "
        "mean, and the NN can learn it without architectural change.",
        "* If `target_std` grows strongly toward low altitudes, and "
        "`|CL_steady - CL_REF|` follows the same trend, the constant "
        "`CL_REF=0.5` is the bottleneck. Mitigation: make `CL_REF` "
        "phase-aware (e.g. `CL_REF(q) = m_ref·g / (q·S_REF)`), so the "
        "NN learns a small correction around the analytical steady-state "
        "rather than the steady-state itself.",
        "",
    ]
    out_path.write_text("\n".join(lines))
    return {
        "median_cl_cruise": pc["median"],
        "p10": pc["p10"],
        "p90": pc["p90"],
        "median_in_band": median_in_band,
        "p10_p90_in_band": p10_p90_in_band,
        "target_cruise_res_p999": tc_res_p999,
        "target_all_res_p999": ta_res_p999,
        "normalizer_p999": normalizer_p999,
        "ratio_normalizer_over_cruise": ratio_norm,
    }


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


@app.command
def parity(
    cl_name: Annotated[
        str, cyclopts.Parameter(help="Sub-directory name for the CL-mode model.")
    ] = "full_hybrid_v3",
    newton_name: Annotated[
        str, cyclopts.Parameter(help="Sub-directory name for the Newton-mode baseline.")
    ] = "full_hybrid_v2",
    config: Annotated[Path, cyclopts.Parameter(help="Pipeline config YAML.")] = Path(
        "config.yaml"
    ),
    device: Annotated[str, cyclopts.Parameter(help="Torch device.")] = "cpu",
    val_limit: Annotated[
        int, cyclopts.Parameter(help="Cap val samples for identifiability_test.")
    ] = 2000,
    factor: Annotated[
        float, cyclopts.Parameter(help="Identifiability perturbation factor.")
    ] = 1.3,
) -> int:
    """AC2/AC3/AC5/AC6 — write `parity_report.md` under the CL model directory."""
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    models_dir = cfg.paths.resolve("models_dir")
    cl_dir = models_dir / cl_name
    newton_dir = models_dir / newton_name

    cl_loss = _load_loss_summary(cl_dir, cl_name)
    newton_loss = _load_loss_summary(newton_dir, newton_name)
    cl_coefs = _load_mass_coefficients(cl_dir)
    newton_coefs = _load_mass_coefficients(newton_dir)
    cl_meta = json.loads((cl_dir / "meta.json").read_text())
    newton_meta = json.loads((newton_dir / "meta.json").read_text())

    log.info("running_identifiability_test", model=cl_name, factor=factor)
    trainer, _, _, _ = _build_trainer_for_model(cl_name, config, device, val_limit)
    identifiability = trainer.identifiability_test(factor=factor)

    out_path = cl_dir / "parity_report.md"
    verdict = _write_parity_report(
        out_path=out_path,
        cl=cl_loss,
        newton=newton_loss,
        cl_coefs=cl_coefs,
        newton_coefs=newton_coefs,
        identifiability=identifiability,
        cl_meta=cl_meta,
        newton_meta=newton_meta,
    )
    log.info("parity_report_written", path=str(out_path), verdict=verdict)
    return 0


@app.command(name="cl-dist")
def cl_dist(
    cl_name: Annotated[
        str, cyclopts.Parameter(help="Sub-directory name for the CL-mode model.")
    ] = "full_hybrid_v3",
    config: Annotated[Path, cyclopts.Parameter(help="Pipeline config YAML.")] = Path(
        "config.yaml"
    ),
    device: Annotated[str, cyclopts.Parameter(help="Torch device.")] = "cpu",
    val_limit: Annotated[
        int, cyclopts.Parameter(help="Cap val samples for the forward pass.")
    ] = 2000,
) -> int:
    """AC4 — write `cl_distribution.md` under the CL model directory."""
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    models_dir = cfg.paths.resolve("models_dir")
    cl_dir = models_dir / cl_name

    log.info("running_cl_distribution", model=cl_name)
    trainer, val_ds, spec, meta = _build_trainer_for_model(cl_name, config, device, val_limit)
    diag = _collect_cl_diagnostics(trainer, val_ds, spec)

    normalizer_stats = (meta.get("stats_dict") or {}).get("fdm_cl_residual")
    out_path = cl_dir / "cl_distribution.md"
    verdict = _write_cl_distribution(out_path, diag, cl_name, normalizer_stats)
    log.info("cl_distribution_written", path=str(out_path), verdict=verdict)
    return 0


@app.command
def run(
    cl_name: Annotated[
        str, cyclopts.Parameter(help="Sub-directory name for the CL-mode model.")
    ] = "full_hybrid_v3",
    newton_name: Annotated[
        str, cyclopts.Parameter(help="Sub-directory name for the Newton-mode baseline.")
    ] = "full_hybrid_v2",
    config: Annotated[Path, cyclopts.Parameter(help="Pipeline config YAML.")] = Path(
        "config.yaml"
    ),
    device: Annotated[str, cyclopts.Parameter(help="Torch device.")] = "cpu",
    val_limit: Annotated[int, cyclopts.Parameter(help="Cap val samples for both passes.")] = 2000,
    factor: Annotated[
        float, cyclopts.Parameter(help="Identifiability perturbation factor.")
    ] = 1.3,
) -> int:
    """Run both `parity` and `cl-dist` and emit the §16 inputs for PHASE_1_RESULTS.md."""
    rc = parity(
        cl_name=cl_name,
        newton_name=newton_name,
        config=config,
        device=device,
        val_limit=val_limit,
        factor=factor,
    )
    if rc != 0:
        return rc
    return cl_dist(cl_name=cl_name, config=config, device=device, val_limit=val_limit)


def main() -> int:
    """Entry point for ``uv run python scripts/phase15_parity_report.py``."""
    return app() or 0


if __name__ == "__main__":
    sys.exit(main())
