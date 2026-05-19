"""AXM-1740 recalibration script — absolute-score identifiability gate.

Loads the two trained models that bracket the Phase 1.5 / Phase 2 debate
(``full_hybrid_v2`` Newton-mode and ``full_hybrid_v3`` CL mass-aware),
runs :meth:`node_fdm.trainer.ODETrainer.identifiability_test_absolute`
on both, calibrates the acceptance floor ``k_min = 0.8 * absolute_score_v2``
(20 % margin around the documented-fonctionnel Newton baseline) and
writes a single markdown artifact summarising setup, sigma_obs, scores,
calibration and verdict.

Usage (from repo root, once both models are trained)::

    uv run python scripts/axm1740_recalibrate.py run \\
        --newton-name full_hybrid_v2 \\
        --cl-name full_hybrid_v3

The artifact is written to
``data/models/axm1740_identifiability_recalibration.md`` and is intended
to be reviewed alongside Phase 1.5 / Phase 2 parity reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import cyclopts
import structlog
from identifiability_diag import format_identifiability_section
from phase15_parity_report import _build_trainer_for_model

log = structlog.get_logger(__name__)

app = cyclopts.App(
    name="axm1740_recalibrate",
    help="AXM-1740 absolute-score identifiability recalibration.",
)

K_CALIBRATION_FRACTION: float = 0.8


def _run_absolute(
    model_name: str,
    config: Path,
    device: str,
    val_limit: int | None,
    factor: float,
    k_min: float | None,
) -> tuple[dict[str, float | bool], dict[str, object]]:
    """Run the absolute-score gate for *model_name* and return ``(result, meta)``."""
    trainer, _, _, meta = _build_trainer_for_model(model_name, config, device, val_limit)
    result = trainer.identifiability_test_absolute(factor=factor, k_min=k_min)
    log.info(
        "absolute_score",
        model=model_name,
        baseline_mse=result["baseline_mse"],
        perturbed_mse=result["perturbed_mse"],
        absolute_score=result["absolute_score"],
        k_min=k_min,
    )
    return result, meta


def _write_recalibration_markdown(
    out_path: Path,
    newton_name: str,
    newton_result: dict[str, float | bool],
    newton_meta: dict[str, object],
    cl_name: str,
    cl_result: dict[str, float | bool],
    cl_meta: dict[str, object],
    k_min: float,
    factor: float,
) -> None:
    """Emit the AXM-1740 recalibration markdown to *out_path*."""
    cl_score = float(cl_result["absolute_score"])
    verdict = "PASS" if cl_score >= k_min else "FAIL"

    lines = [
        "# AXM-1740 — Absolute-Score Identifiability Recalibration",
        "",
        "## Setup",
        "",
        "| Field | Value |",
        "| -- | -- |",
        f"| newton_model | {newton_name} ({newton_meta.get('architecture_name')}) |",
        f"| cl_model | {cl_name} ({cl_meta.get('architecture_name')}) |",
        f"| factor | {factor:.3f} |",
        f"| k_calibrated | {K_CALIBRATION_FRACTION:.2f} x absolute_score_v2 |",
        f"| k_min | {k_min:.6f} |",
        "",
        format_identifiability_section(newton_result, newton_name, k_min=k_min),
        format_identifiability_section(cl_result, cl_name, k_min=k_min),
        "## Verdict",
        "",
        f"* `{cl_name}` absolute_score = {cl_score:.6f}",
        f"* k_min = {k_min:.6f}",
        f"* **{verdict}** — "
        + (
            "CL mass-aware identifiability preserved despite degraded relative ratio."
            if verdict == "PASS"
            else "real identifiability loss — distinct from the AC5 ratio false-negative."
        ),
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    log.info("recalibration_artifact_written", path=str(out_path), verdict=verdict)


@app.command
def run(
    newton_name: Annotated[
        str, cyclopts.Parameter(help="Sub-directory name for the Newton-mode baseline.")
    ] = "full_hybrid_v2",
    cl_name: Annotated[
        str, cyclopts.Parameter(help="Sub-directory name for the CL-mode model.")
    ] = "full_hybrid_v3",
    config: Annotated[Path, cyclopts.Parameter(help="Pipeline config YAML.")] = Path(
        "config.yaml"
    ),
    device: Annotated[str, cyclopts.Parameter(help="Torch device.")] = "cpu",
    val_limit: Annotated[
        int, cyclopts.Parameter(help="Cap val samples for the identifiability pass.")
    ] = 2000,
    factor: Annotated[
        float, cyclopts.Parameter(help="Identifiability perturbation factor on m_0.")
    ] = 1.3,
) -> int:
    """AC3/AC4/AC5 — write `axm1740_identifiability_recalibration.md`."""
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(config)
    models_dir = cfg.paths.resolve("models_dir")

    log.info("running_absolute_score", model=newton_name, factor=factor)
    newton_result, newton_meta = _run_absolute(
        newton_name, config, device, val_limit, factor, k_min=None
    )
    newton_score = float(newton_result["absolute_score"])
    k_min = K_CALIBRATION_FRACTION * newton_score

    log.info("running_absolute_score", model=cl_name, factor=factor, k_min=k_min)
    cl_result, cl_meta = _run_absolute(cl_name, config, device, val_limit, factor, k_min=k_min)

    out_path = models_dir / "axm1740_identifiability_recalibration.md"
    _write_recalibration_markdown(
        out_path=out_path,
        newton_name=newton_name,
        newton_result=newton_result,
        newton_meta=newton_meta,
        cl_name=cl_name,
        cl_result=cl_result,
        cl_meta=cl_meta,
        k_min=k_min,
        factor=factor,
    )
    # Also emit a JSON dump for downstream programmatic consumers.
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "newton_name": newton_name,
                "cl_name": cl_name,
                "factor": factor,
                "k_min": k_min,
                "newton": newton_result,
                "cl": cl_result,
            },
            indent=2,
            default=float,
        )
    )
    return 0


if __name__ == "__main__":
    app()
