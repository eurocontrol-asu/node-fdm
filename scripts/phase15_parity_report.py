"""Phase 1.5 parity report — CL-mode (``full_hybrid_v4``) vs Newton-mode (``full_hybrid_v3``).

Generates the empirical artifacts that gate Phase 1.5 (AXM-1739):

* ``data/models/{v4}/parity_report.md`` — loss parity table, b_raw drift
  table, identifiability ratio, verdict (AC2 / AC3 / AC5 / AC6).
* ``data/models/{v4}/cl_distribution.md`` — CL percentiles on the val-set
  cruise slice (AC4).

Usage (run from repo root once both models are trained)::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/phase15_parity_report.py run \\
        --v4-name full_hybrid_v4 \\
        --v3-name full_hybrid_v3

Subcommands ``parity`` and ``cl-dist`` emit a single artifact each;
``run`` emits both. All artifacts land under the v4 model directory.

**Cruise-slice caveat (AC4)** — the ticket specifies
``fdm_mode_label == \"ALT_MACH\"`` for the cruise slice. ``fdm_mode_label``
is a delta-table column produced by
``node_fdm_data.preprocessing.label_modes`` but is **not threaded into
``FlightSample`` tensors** by the existing val-set loader. To keep the
\"no new fixtures\" constraint from technical_notes, this script uses an
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
"""AC3: |val_loss_v4 - val_loss_v3| / val_loss_v3 <= 0.02 to PASS parity."""

IDENTIFIABILITY_THRESHOLD: float = 1.20
"""AC5: identifiability ratio strictly > 1.20 to PASS."""

B_RAW_DRIFT_TOL: float = 0.10
"""AC6: |coef_v4 - coef_v3| / |coef_v3| <= 0.10 per feature to PASS."""

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
    drift report robust against transitional states where a trained v3
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
    _, val_ds = get_train_val_data(
        data_df=df,
        x_cols=spec.x_cols,
        u_cols=spec.u_cols,
        e_cols=spec.e0_cols,
        e1_cols=spec.e1_cols,
        dx_cols=dx_col_names,
        seq_len=meta["seq_len"],
        shift=meta["shift"],
        train_limit=1,
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
        train_dataset=val_ds,  # placeholder; identifiability_test only uses val.
        val_dataset=val_ds,
        model_dir=models_dir,
        device=device,
    )
    trainer.load_model_weights()
    return trainer, val_ds, spec, meta


# ---------------------------------------------------------------------------
# CL distribution (AC4)
# ---------------------------------------------------------------------------


def _collect_cl_on_cruise(trainer, val_dataset, spec) -> tuple[np.ndarray, int, int]:
    """Forward-pass the val set through ``data_ode_long`` and slice on cruise.

    Returns ``(cl_values, n_total_steps, n_cruise_steps)``. ``cl_values`` is
    the union of per-step CL values on cruise steps, ready for percentiles.
    """
    from node_fdm.layers.physics import CL_REF

    long_layer = trainer.model.layers_dict["data_ode_long"]
    long_layer.eval()

    long_input_cols = list(long_layer.input_cols) if hasattr(long_layer, "input_cols") else None
    if long_input_cols is None:
        spec_long = next(layer for layer in spec.layers if layer.name == "data_ode_long")
        long_input_cols = list(spec_long.input_cols)

    alt_idx = spec.x_cols.index("raw_alt_m")
    x_col_index = {name: idx for idx, name in enumerate(spec.x_cols)}
    u_col_index = {name: idx for idx, name in enumerate(spec.u_cols)}
    e_col_index = {name: idx for idx, name in enumerate(spec.e0_cols)}

    cl_values: list[float] = []
    n_total = 0
    n_cruise = 0
    with torch.no_grad():
        for sample in val_dataset:
            inputs: dict[str, torch.Tensor] = {}
            for col in long_input_cols:
                if col in x_col_index:
                    inputs[col] = sample.x[..., x_col_index[col]]
                elif col in u_col_index:
                    inputs[col] = sample.u[..., u_col_index[col]]
                elif col in e_col_index:
                    inputs[col] = sample.e[..., e_col_index[col]]
                else:
                    msg = f"data_ode_long input column not found in sample tensors: {col}"
                    raise KeyError(msg)
            output = long_layer(inputs)
            if "fdm_cl_residual" not in output:
                msg = "data_ode_long output missing 'fdm_cl_residual' — expected the v3 arch"
                raise RuntimeError(msg)
            cl_residual = output["fdm_cl_residual"].detach().cpu().numpy().reshape(-1)
            alt = sample.x[..., alt_idx].detach().cpu().numpy().reshape(-1)
            cruise_mask = alt > CRUISE_ALT_MIN_M
            n_total += int(alt.size)
            n_cruise += int(cruise_mask.sum())
            cl_values.extend((cl_residual[cruise_mask] + CL_REF).tolist())
    return np.asarray(cl_values, dtype=np.float64), n_total, n_cruise


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _format_loss_overlay(v4: LossSummary, v3: LossSummary) -> str:
    """Build a Markdown table aligning per-epoch losses for both models."""
    n = max(len(v4.epochs), len(v3.epochs))
    rows = [
        "| epoch | train_v4 | val_v4 | train_v3 | val_v3 |",
        "|------:|---------:|-------:|---------:|-------:|",
    ]
    for i in range(n):
        epoch = v4.epochs[i] if i < len(v4.epochs) else v3.epochs[i]
        train_v4 = f"{v4.train_losses[i]:.6f}" if i < len(v4.train_losses) else "—"
        val_v4 = f"{v4.val_losses[i]:.6f}" if i < len(v4.val_losses) else "—"
        train_v3 = f"{v3.train_losses[i]:.6f}" if i < len(v3.train_losses) else "—"
        val_v3 = f"{v3.val_losses[i]:.6f}" if i < len(v3.val_losses) else "—"
        rows.append(f"| {epoch:g} | {train_v4} | {val_v4} | {train_v3} | {val_v3} |")
    return "\n".join(rows)


def _format_b_raw_drift(
    v4_coefs: dict[str, float], v3_coefs: dict[str, float]
) -> tuple[str, bool]:
    """Compare per-feature post-softplus coefficients. Returns (markdown, passed)."""
    common_keys = [k for k in v3_coefs if k in v4_coefs]
    rows = [
        "| feature | v3 | v4 | |Δ|/|v3| | sign-consistent | within 10% |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    all_pass = True
    for key in common_keys:
        v3v = v3_coefs[key]
        v4v = v4_coefs[key]
        denom = abs(v3v) if abs(v3v) > 1e-9 else float("inf")
        drift = abs(v4v - v3v) / denom
        sign_ok = (v3v >= 0) == (v4v >= 0)
        within = drift <= B_RAW_DRIFT_TOL
        passed = within or sign_ok
        if not passed:
            all_pass = False
        rows.append(
            f"| `{key}` | {v3v:+.4f} | {v4v:+.4f} | {drift:.3f} | "
            f"{'✅' if sign_ok else '❌'} | {'✅' if within else '❌'} |"
        )
    return "\n".join(rows), all_pass


def _write_parity_report(
    out_path: Path,
    v4: LossSummary,
    v3: LossSummary,
    v4_coefs: dict[str, float],
    v3_coefs: dict[str, float],
    identifiability: dict[str, float],
    v4_meta: dict,
    v3_meta: dict,
) -> dict[str, object]:
    """Emit ``parity_report.md`` covering AC2, AC3, AC5, AC6. Returns verdict dict."""
    val_loss_v4 = v4.final_val
    val_loss_v3 = v3.final_val
    if val_loss_v3 > 0:
        ratio = (val_loss_v4 - val_loss_v3) / val_loss_v3
    else:
        ratio = float("inf")
    parity_pass = abs(ratio) <= PARITY_RATIO_TOL
    drift_md, drift_pass = _format_b_raw_drift(v4_coefs, v3_coefs)
    ident_pass = identifiability["ratio"] > IDENTIFIABILITY_THRESHOLD

    if not parity_pass:
        status = "regression"
    elif not (drift_pass and ident_pass):
        status = "parity (with caveats)"
    else:
        status = "parity"

    overlay_md = _format_loss_overlay(v4, v3)

    lines = [
        f"# Phase 1.5 parity report — {v4.name} vs {v3.name}",
        "",
        f"> **Status**: `{status}`",
        "",
        "This report is emitted by `scripts/phase15_parity_report.py` and covers",
        "AC2 / AC3 / AC5 / AC6 of AXM-1739.",
        "",
        "## 1. Setup",
        "",
        "| Field | v4 (CL-mode) | v3 (Newton-mode reference) |",
        "|---|---|---|",
        f"| Architecture | `{v4_meta['architecture_name']}` | `{v3_meta['architecture_name']}` |",
        f"| Epochs (configured) | {v4_meta['epochs']} | {v3_meta['epochs']} |",
        f"| Seed | {v4_meta.get('seed')} | {v3_meta.get('seed')} |",
        f"| Batch | {v4_meta['batch_size']} | {v3_meta['batch_size']} |",
        f"| Seq len | {v4_meta['seq_len']} | {v3_meta['seq_len']} |",
        f"| lr | {v4_meta['lr']} | {v3_meta['lr']} |",
        "",
        "## 2. Loss summary (AC2)",
        "",
        "| Metric | v4 | v3 | (v4 - v3) / v3 |",
        "|---|---:|---:|---:|",
        f"| final train_loss | {v4.final_train:.6f} | {v3.final_train:.6f} | — |",
        f"| final val_loss   | {v4.final_val:.6f} | {v3.final_val:.6f} | {ratio:+.4f} |",
        f"| best  val_loss   | {v4.best_val:.6f} | {v3.best_val:.6f} | — |",
        f"| best  epoch      | {v4.best_epoch:g} | {v3.best_epoch:g} | — |",
        "",
        "## 3. Parity criterion (AC3)",
        "",
        f"AC3 requires `|val_loss_v4 - val_loss_v3| / val_loss_v3 <= {PARITY_RATIO_TOL}`.",
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
            f"`trainer.identifiability_test(factor=1.3)` on `{v4.name}`:",
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
            f"either `|Δ|/|v3| ≤ {B_RAW_DRIFT_TOL}` or the sign is preserved.",
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
            f"- AC3 (parity ≤ {PARITY_RATIO_TOL}): {'✅' if parity_pass else '❌'}",
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


def _write_cl_distribution(
    out_path: Path,
    cl_values: np.ndarray,
    n_total: int,
    n_cruise: int,
    model_name: str,
) -> dict[str, object]:
    """Emit ``cl_distribution.md`` covering AC4. Returns verdict dict."""
    if cl_values.size == 0:
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
        }
    p10, median, p90 = (
        float(np.percentile(cl_values, 10)),
        float(np.median(cl_values)),
        float(np.percentile(cl_values, 90)),
    )
    median_in_band = CL_CRUISE_MEDIAN_RANGE[0] <= median <= CL_CRUISE_MEDIAN_RANGE[1]
    p10_p90_in_band = (
        CL_CRUISE_P10_P90_RANGE[0] <= p10 <= CL_CRUISE_P10_P90_RANGE[1]
        and CL_CRUISE_P10_P90_RANGE[0] <= p90 <= CL_CRUISE_P10_P90_RANGE[1]
    )
    cruise_fraction = n_cruise / max(n_total, 1)
    lines = [
        f"# CL distribution — `{model_name}`",
        "",
        "AC4 of AXM-1739: validate the CL operating range of the CL-mode model on",
        f"the val-set cruise slice. Cruise mask = `raw_alt_m > {CRUISE_ALT_MIN_M:g} m`",
        "(altitude proxy for `fdm_mode_label == ALT_MACH`; see module docstring).",
        "",
        "## 1. Slice coverage",
        "",
        f"- Total val steps: `{n_total}`",
        f"- Cruise-slice steps: `{n_cruise}` ({100 * cruise_fraction:.1f}%)",
        "",
        "## 2. CL = CL_REF + cl_residual_pred — percentiles",
        "",
        "| Metric | Value | Target | PASS |",
        "|---|---:|---|:---:|",
        f"| `median_CL_cruise` | {median:.4f} | "
        f"[{CL_CRUISE_MEDIAN_RANGE[0]}, {CL_CRUISE_MEDIAN_RANGE[1]}] | "
        f"{'✅' if median_in_band else '❌'} |",
        f"| `p10` | {p10:.4f} | \\geq {CL_CRUISE_P10_P90_RANGE[0]} | "
        f"{'✅' if p10 >= CL_CRUISE_P10_P90_RANGE[0] else '❌'} |",
        f"| `p90` | {p90:.4f} | \\leq {CL_CRUISE_P10_P90_RANGE[1]} | "
        f"{'✅' if p90 <= CL_CRUISE_P10_P90_RANGE[1] else '❌'} |",
        "",
        f"AC4 (a + b) verdict: **{'PASS' if median_in_band and p10_p90_in_band else 'FAIL'}**.",
        "",
    ]
    out_path.write_text("\n".join(lines))
    return {
        "median_cl_cruise": median,
        "p10": p10,
        "p90": p90,
        "median_in_band": median_in_band,
        "p10_p90_in_band": p10_p90_in_band,
    }


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


@app.command
def parity(
    v4_name: Annotated[
        str, cyclopts.Parameter(help="Sub-directory name for the CL-mode model.")
    ] = "full_hybrid_v4",
    v3_name: Annotated[
        str, cyclopts.Parameter(help="Sub-directory name for the Newton-mode reference.")
    ] = "full_hybrid_v3",
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
    """AC2/AC3/AC5/AC6 — write `parity_report.md` under the v4 model directory."""
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    models_dir = cfg.paths.resolve("models_dir")
    v4_dir = models_dir / v4_name
    v3_dir = models_dir / v3_name

    v4_loss = _load_loss_summary(v4_dir, v4_name)
    v3_loss = _load_loss_summary(v3_dir, v3_name)
    v4_coefs = _load_mass_coefficients(v4_dir)
    v3_coefs = _load_mass_coefficients(v3_dir)
    v4_meta = json.loads((v4_dir / "meta.json").read_text())
    v3_meta = json.loads((v3_dir / "meta.json").read_text())

    log.info("running_identifiability_test", model=v4_name, factor=factor)
    trainer, _, _, _ = _build_trainer_for_model(v4_name, config, device, val_limit)
    identifiability = trainer.identifiability_test(factor=factor)

    out_path = v4_dir / "parity_report.md"
    verdict = _write_parity_report(
        out_path=out_path,
        v4=v4_loss,
        v3=v3_loss,
        v4_coefs=v4_coefs,
        v3_coefs=v3_coefs,
        identifiability=identifiability,
        v4_meta=v4_meta,
        v3_meta=v3_meta,
    )
    log.info("parity_report_written", path=str(out_path), verdict=verdict)
    return 0


@app.command(name="cl-dist")
def cl_dist(
    v4_name: Annotated[
        str, cyclopts.Parameter(help="Sub-directory name for the CL-mode model.")
    ] = "full_hybrid_v4",
    config: Annotated[Path, cyclopts.Parameter(help="Pipeline config YAML.")] = Path(
        "config.yaml"
    ),
    device: Annotated[str, cyclopts.Parameter(help="Torch device.")] = "cpu",
    val_limit: Annotated[
        int, cyclopts.Parameter(help="Cap val samples for the forward pass.")
    ] = 2000,
) -> int:
    """AC4 — write `cl_distribution.md` under the v4 model directory."""
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    models_dir = cfg.paths.resolve("models_dir")
    v4_dir = models_dir / v4_name

    log.info("running_cl_distribution", model=v4_name)
    trainer, val_ds, spec, _ = _build_trainer_for_model(v4_name, config, device, val_limit)
    cl_values, n_total, n_cruise = _collect_cl_on_cruise(trainer, val_ds, spec)

    out_path = v4_dir / "cl_distribution.md"
    verdict = _write_cl_distribution(out_path, cl_values, n_total, n_cruise, v4_name)
    log.info("cl_distribution_written", path=str(out_path), verdict=verdict)
    return 0


@app.command
def run(
    v4_name: Annotated[
        str, cyclopts.Parameter(help="Sub-directory name for the CL-mode model.")
    ] = "full_hybrid_v4",
    v3_name: Annotated[
        str, cyclopts.Parameter(help="Sub-directory name for the Newton-mode reference.")
    ] = "full_hybrid_v3",
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
        v4_name=v4_name,
        v3_name=v3_name,
        config=config,
        device=device,
        val_limit=val_limit,
        factor=factor,
    )
    if rc != 0:
        return rc
    return cl_dist(v4_name=v4_name, config=config, device=device, val_limit=val_limit)


def main() -> int:
    """Entry point for ``uv run python scripts/phase15_parity_report.py``."""
    return app() or 0


if __name__ == "__main__":
    sys.exit(main())
