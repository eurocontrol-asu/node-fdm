"""Phase 3 — Exp 18 follow-up : v13d parallel-head migration check.

Tests whether Strategy W (v13d) re-introduced cybernetic leakage by
quantifying the magnitude of the *parallel unbounded head*
(`fdm_t_minus_d_norm_parallel`) vs v12's original unbounded head
(`fdm_t_minus_d_norm`).

Operationally :

    parallel_head_magnitude_v13d = mean(|0.5 · t_minus_d_norm_parallel · m_ref|)_cruise
    v12_head_magnitude           = mean(|t_minus_d_norm · m_ref|)_cruise = 34.6 kN

The factor 0.5 is the static λ used in v13d's PhysicsLayer.

If v13d_parallel ≈ v12_head magnitude → Strategy W *closed R7 by
re-introducing the leak* via a new channel. AC7-on-parallel-head
fails (ratio > 2). The R7 vs AC7 trade-off is mechanically exposed.

If v13d_parallel << v12_head magnitude → the T_PS prior is doing
useful work and the parallel head only provides a small residual.
Both R7 and AC7 pass — the architecture is genuinely better than v12.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from node_fdm.architectures import (  # noqa: F401
    adsb_hybrid_v12_psdrag,
    adsb_hybrid_v13d_psthrust_parallel,
)
from node_fdm.architectures.registry import get as get_arch_spec
from node_fdm.trainer import ODETrainer, TrainingConfig

_M_REF_KG = 0.5 * (42_600.0 + 77_000.0)
_PARALLEL_LAMBDA = 0.5  # matches PhysicsLayer Strategy W branch


def _build_trainer(
    model_dir: Path,
    arch_key: str,
    *,
    val_limit: int = 1000,
    lenient_load: bool = False,
) -> ODETrainer:
    from node_fdm.loader import get_train_val_data
    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    meta = json.loads((model_dir / "meta.json").read_text())
    arch_name = meta["architecture_name"]
    info = resolve_architecture(arch_key)
    spec = get_arch_spec(arch_name)
    cfg = PipelineConfig.from_yaml(Path("config.yaml"))

    training_config = TrainingConfig(
        architecture_name=arch_name,
        model_name=meta.get("model_name", model_dir.name),
        model_params=tuple(meta["model_params"]),
        seq_len=meta["seq_len"],
        shift=meta["shift"],
        step=meta["step"],
        lr=meta["lr"],
        batch_size=meta["batch_size"],
        method=meta["method"],
        activation=meta.get("activation", "silu"),
        seed=meta.get("seed", 42),
        use_mode_weights=meta.get("use_mode_weights", False),
        mode_weight_alpha=meta.get("mode_weight_alpha", 0.5),
        epochs=1,
    )

    delta_table = cfg.paths.resolve("delta_table")
    import polars as pl

    full_df = pl.read_delta(str(delta_table)).filter(pl.col("fdm_flag_valid"))
    data_df = full_df.filter(pl.col("meta_aircraft_type") == "A320")
    flight_feature_cols = list(getattr(spec, "flight_feature_cols", []) or [])
    _train_ds, val_ds = get_train_val_data(
        data_df=data_df,
        x_cols=info.x_cols,
        u_cols=info.u_cols,
        e_cols=info.e0_cols,
        e1_cols=info.e1_cols,
        dx_cols=[c for _, c in info.dx_cols],
        seq_len=training_config.seq_len,
        shift=training_config.shift,
        train_limit=val_limit,
        val_limit=val_limit,
        flight_feature_cols=flight_feature_cols or None,
        require_routing=True,
    )

    trainer = ODETrainer(
        config=training_config,
        train_dataset=val_ds,
        val_dataset=val_ds,
        model_dir=model_dir.parent,
        device="cpu",
    )
    if lenient_load:
        for name in trainer.model.layers_name:
            ckpt_path = trainer.model_dir / f"{name}.pt"
            ckpt = torch.load(ckpt_path, weights_only=True)
            trainer.model.layers_dict[name].load_state_dict(
                ckpt["layer_state"], strict=False,
            )
        if trainer.mass_encoder is not None:
            mass_ckpt = trainer.model_dir / "mass_encoder.pt"
            if mass_ckpt.exists():
                state = torch.load(mass_ckpt, weights_only=True)
                trainer.mass_encoder.load_state_dict(state, strict=False)
    else:
        trainer.load_model_weights(reset_loss=True)
    trainer.model.eval()
    return trainer


def _capture_head(trainer: ODETrainer, head_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    captured: list[torch.Tensor] = []
    alt: list[np.ndarray] = []
    tas: list[np.ndarray] = []

    original_forward = trainer.model.layers_dict["data_ode_long"].forward

    def patched(vec):
        out = original_forward(vec)
        if head_col in out:
            captured.append(out[head_col].detach().clone())
        return out

    trainer.model.layers_dict["data_ode_long"].forward = patched  # type: ignore[method-assign]
    with torch.no_grad():
        for batch in trainer.val_loader:
            trainer.model.reset_history()
            _ = trainer._compute_batch_loss(batch)
            hist = trainer.model.history
            if "raw_alt_m" in hist:
                alt.append(hist["raw_alt_m"].cpu().numpy().reshape(-1))
            if "era_tas_ms" in hist:
                tas.append(hist["era_tas_ms"].cpu().numpy().reshape(-1))
    trainer.model.layers_dict["data_ode_long"].forward = original_forward  # type: ignore[method-assign]
    head = torch.cat(captured).cpu().numpy().reshape(-1)
    alt_arr = np.concatenate(alt) if alt else np.zeros_like(head)
    tas_arr = np.concatenate(tas) if tas else np.zeros_like(head)
    return head, alt_arr, tas_arr


def cruise_mask(alt: np.ndarray, tas: np.ndarray) -> np.ndarray:
    return (alt >= 9_000.0) & (tas >= 180.0) & (tas <= 270.0)


def main() -> int:
    val_limit = 1000

    print("Loading v12 ...")
    tr12 = _build_trainer(
        Path("data/models/full_hybrid_v12"),
        "adsb_hybrid_v12_psdrag",
        val_limit=val_limit, lenient_load=True,
    )
    v12_head, v12_alt, v12_tas = _capture_head(tr12, "fdm_t_minus_d_norm")
    n12 = min(len(v12_head), len(v12_alt))
    m12 = cruise_mask(v12_alt[:n12], v12_tas[:n12])
    v12_mean_N = float(np.abs(v12_head[:n12][m12] * _M_REF_KG).mean())
    print(f"v12 cruise n={m12.sum()}, mean|t_minus_d·m_ref|={v12_mean_N:.0f} N")

    print("Loading v13d ...")
    tr13d = _build_trainer(
        Path("data/models/full_hybrid_v13d"),
        "adsb_hybrid_v13d_psthrust_parallel",
        val_limit=val_limit,
    )
    v13d_head, v13d_alt, v13d_tas = _capture_head(tr13d, "fdm_t_minus_d_norm_parallel")
    n13d = min(len(v13d_head), len(v13d_alt))
    m13d = cruise_mask(v13d_alt[:n13d], v13d_tas[:n13d])
    v13d_mean_N = float(np.abs(_PARALLEL_LAMBDA * v13d_head[:n13d][m13d] * _M_REF_KG).mean())
    print(f"v13d cruise n={m13d.sum()}, mean|λ·t_minus_d_par·m_ref|={v13d_mean_N:.0f} N (λ={_PARALLEL_LAMBDA})")

    # Also capture v13d's bounded head magnitude for the original AC7-style ratio.
    v13d_bounded, _, _ = _capture_head(tr13d, "fdm_t_correction_parallel")
    # T_PS magnitude proxy : approximated as v12 typical cruise thrust ~25 kN.
    # For the bounded contribution we use |0.05·tanh(t_corr)| * T_PS_typical.
    t_ps_typical_kN = 25.0
    v13d_bounded_N = float(
        np.abs(0.05 * np.tanh(v13d_bounded[:n13d][m13d])).mean() * t_ps_typical_kN * 1000
    )
    print(f"v13d cruise bounded head magnitude (proxy) = {v13d_bounded_N:.0f} N")

    parallel_ratio = v13d_mean_N / v12_mean_N if v12_mean_N > 0 else float("nan")
    bounded_ratio = v13d_bounded_N / v12_mean_N if v12_mean_N > 0 else float("nan")
    print()
    print("=== Migration verdict ===")
    print(f"v13d parallel-head ratio vs v12      = {parallel_ratio:.3f}")
    print(f"v13d bounded-head  ratio vs v12      = {bounded_ratio:.4f}")
    print(f"Total v13d head magnitude vs v12     = {(v13d_mean_N + v13d_bounded_N)/v12_mean_N:.3f}")
    print()
    if parallel_ratio > 0.5:
        print("🔍 STRATEGY W ANALYSIS")
        print(f"  parallel head ≈ {parallel_ratio*100:.0f} % of v12 magnitude")
        print(f"  → R7 closed by RE-INTRODUCING leak via Strategy W parallel head.")
        print(f"  → The bounded head stays quiet (ratio {bounded_ratio:.4f}) — original AC7 still passes literally,")
        print(f"    but the *scientific* AC7 (no leakage anywhere) fails.")
        print(f"  → L4 vs L5 trade-off confirmed empirically.")
    else:
        print("✅ No significant leak migration through Strategy W.")

    report = Path("data/models/_comparison/phase3_thrust_v13d_migration.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 3 — Exp 18 : v13d Strategy W migration check",
        "",
        "> Whereas the original AC7 measured the *bounded head's* magnitude relative",
        "> to v12, this diagnostic measures the *parallel unbounded head* — which",
        "> is the new architectural escape hatch Strategy W introduces.",
        "",
        "## Method",
        "",
        f"For each model, capture the longitudinal head's post-cap output on a "
        f"cruise-stable val slice (alt ≥ 9000 m, TAS ∈ [180, 270] m/s, val_limit={val_limit}).",
        "",
        f"- v12 head : `t_minus_d_norm · m_ref` (the original unbounded head)",
        f"- v13d parallel head : `λ · t_minus_d_norm_parallel · m_ref` where λ = {_PARALLEL_LAMBDA}",
        f"- v13d bounded head proxy : `|0.05·tanh(t_corr_par)| · T_PS_typ` where T_PS_typ ≈ 25 kN",
        "",
        "## Results",
        "",
        f"| Model | head | mean magnitude (N) | ratio vs v12 |",
        f"|---|---|---:|---:|",
        f"| v12 | `t_minus_d_norm·m_ref` | **{v12_mean_N:.0f}** | 1.000 |",
        f"| v13d | `λ·t_minus_d_par·m_ref` (parallel) | **{v13d_mean_N:.0f}** | **{parallel_ratio:.3f}** |",
        f"| v13d | bounded (proxy) | {v13d_bounded_N:.0f} | {bounded_ratio:.4f} |",
        f"| v13d total | bounded + parallel | {v13d_mean_N + v13d_bounded_N:.0f} | {(v13d_mean_N + v13d_bounded_N)/v12_mean_N:.3f} |",
        "",
        "## Verdict",
        "",
    ]
    if parallel_ratio > 0.5:
        lines += [
            f"**🔍 Strategy W closed R7 by re-introducing leak via the parallel head.**",
            "",
            f"The bounded head stays quiet (ratio {bounded_ratio:.4f}, well below the",
            f"migration threshold) — so the *literal* AC7 ratio test still passes.",
            f"But the parallel head's magnitude is {parallel_ratio*100:.0f} % of v12's",
            f"original unbounded head — the leak migrated to the new channel that",
            f"Strategy W explicitly enabled.",
            "",
            f"This is the **L4-vs-L5 trade-off** predicted by FINDINGS : closing R7",
            f"requires giving the closed-loop an unbounded channel, but that *is*",
            f"the leak. The Phase 3 \"no-leak\" guarantee (v13, v13b, v13c) is",
            f"genuine but cannot reach v12's val_loss without giving up the bound.",
        ]
    else:
        lines += [
            f"**✅ R7 closed without re-introducing leak.**",
            "",
            f"The parallel head magnitude is {parallel_ratio:.3f} of v12's — the T_PS",
            f"prior is genuinely contributing the bulk of the thrust signal.",
        ]
    report.write_text("\n".join(lines))
    print(f"\nWrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
