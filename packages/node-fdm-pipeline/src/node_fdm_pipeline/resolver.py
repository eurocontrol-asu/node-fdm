"""Architecture resolver — dispatches ``--arch`` flag to schema + preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ARCH_BY_NAME",
    "ArchitectureInfo",
    "resolve_architecture",
]


@dataclass(frozen=True)
class ArchitectureInfo:
    """Resolved architecture with schema columns and preprocessing functions.

    Attributes:
        name: Architecture registry name (e.g. ``"node_adsb_v1"``).
        x_cols: State column names.
        u_cols: Control column names.
        e0_cols: Environment column names.
        e1_cols: Derived environment column names (computed by physics layers).
        dx_cols: Derivative column specs ``(sign, col_name)``.
        preprocessing_fn: Flight preprocessing function (used by predict/stats).
        segment_filter_fn: Optional segment filter function (used by predict/stats).
        architecture_import: Dotted import path to trigger auto-registration.
    """

    name: str
    x_cols: list[str]
    u_cols: list[str]
    e0_cols: list[str]
    dx_cols: list[tuple[int, str]]
    preprocessing_fn: Any
    segment_filter_fn: Any
    architecture_import: str
    e1_cols: list[str] = field(default_factory=list)


#: Reverse mapping from architecture registry name to CLI arch key.
ARCH_BY_NAME: dict[str, str] = {
    "qar": "qar",
    "node_adsb_v1": "adsb",
    "node_adsb_hybrid_v1": "adsb_hybrid",
    "node_adsb_hybrid_v2": "adsb_hybrid_v2",
    "node_adsb_hybrid_v3": "adsb_hybrid_v3",
    "node_adsb_hybrid_v4_mass_features": "adsb_hybrid_v4",
    "node_adsb_hybrid_v5_tempered_t3": "adsb_hybrid_v5_tempered",
    "node_adsb_hybrid_v6_mlp_monotone": "adsb_hybrid_v6_mlp",
    "node_adsb_hybrid_v7_lean_2features": "adsb_hybrid_v7_lean",
    "node_adsb_hybrid_v8_lean_t15": "adsb_hybrid_v8_lean_t15",
    "node_adsb_hybrid_v9_causal_3features": "adsb_hybrid_v9_causal",
    "node_adsb_hybrid_v10_ps_auxloss": "adsb_hybrid_v10_ps_auxloss",
    "node_adsb_hybrid_v11_ps_residual": "adsb_hybrid_v11_ps_residual",
    "node_adsb_hybrid_v12_psdrag": "adsb_hybrid_v12_psdrag",
    "node_adsb_hybrid_v13_psthrust": "adsb_hybrid_v13_psthrust",
    "node_adsb_hybrid_v13b_psthrust_w10": "adsb_hybrid_v13b_psthrust_w10",
    "node_adsb_hybrid_v13c_psthrust_tet": "adsb_hybrid_v13c_psthrust_tet",
    "node_adsb_hybrid_v13d_psthrust_parallel": "adsb_hybrid_v13d_psthrust_parallel",
    "node_adsb_hybrid_v13e_psthrust_parallel_l025": "adsb_hybrid_v13e_psthrust_parallel_l025",
    "node_adsb_hybrid_v13f_psthrust_parallel_anneal": "adsb_hybrid_v13f_psthrust_parallel_anneal",
    "node_adsb_hybrid_v13g_psthrust_tet_parallel": "adsb_hybrid_v13g_psthrust_tet_parallel",
    "node_adsb_hybrid_v14_psefficiency": "adsb_hybrid_v14_psefficiency",
}

_SUPPORTED_ARCHS: tuple[str, ...] = (
    "qar",
    "adsb",
    "adsb_hybrid",
    "adsb_hybrid_v2",
    "adsb_hybrid_v3",
    "adsb_hybrid_v4",
    "adsb_hybrid_v5_tempered",
    "adsb_hybrid_v6_mlp",
    "adsb_hybrid_v7_lean",
    "adsb_hybrid_v8_lean_t15",
    "adsb_hybrid_v9_causal",
    "adsb_hybrid_v10_ps_auxloss",
    "adsb_hybrid_v11_ps_residual",
    "adsb_hybrid_v12_psdrag",
    "adsb_hybrid_v13_psthrust",
    "adsb_hybrid_v13b_psthrust_w10",
    "adsb_hybrid_v13c_psthrust_tet",
    "adsb_hybrid_v13d_psthrust_parallel",
    "adsb_hybrid_v13e_psthrust_parallel_l025",
    "adsb_hybrid_v13f_psthrust_parallel_anneal",
    "adsb_hybrid_v13g_psthrust_tet_parallel",
    "adsb_hybrid_v14_psefficiency",
)


def resolve_architecture(arch: str) -> ArchitectureInfo:
    """Resolve an architecture name to its schema and preprocessing components.

    Args:
        arch: Architecture identifier — one of ``"qar"``, ``"adsb"``,
            ``"adsb_hybrid"`` (v1, Newton 5 features),
            ``"adsb_hybrid_v2"`` (Newton 6 features, Phase 1 baseline),
            ``"adsb_hybrid_v3"`` (CL-mode, Phase 1.5).

    Returns:
        Fully resolved ``ArchitectureInfo``.

    Raises:
        ValueError: If *arch* is not a supported architecture.
    """
    match arch:
        case "qar":
            from node_fdm_data.schemas.qar import DX_COLS, E0_COLS, E1_COLS, U_COLS, X_COLS

            return ArchitectureInfo(
                name="qar",
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                e1_cols=E1_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=None,
                segment_filter_fn=None,
                architecture_import="node_fdm.architectures.qar",
            )
        case "adsb":
            from node_fdm_data.schemas.adsb import DX_COLS, E0_COLS, E1_COLS, U_COLS, X_COLS

            return ArchitectureInfo(
                name="node_adsb_v1",
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                e1_cols=E1_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=None,
                segment_filter_fn=None,
                architecture_import="node_fdm.architectures.adsb",
            )
        case (
            "adsb_hybrid"
            | "adsb_hybrid_v2"
            | "adsb_hybrid_v3"
            | "adsb_hybrid_v4"
            | "adsb_hybrid_v5_tempered"
            | "adsb_hybrid_v6_mlp"
            | "adsb_hybrid_v7_lean"
            | "adsb_hybrid_v8_lean_t15"
            | "adsb_hybrid_v9_causal"
            | "adsb_hybrid_v10_ps_auxloss"
            | "adsb_hybrid_v11_ps_residual"
            | "adsb_hybrid_v12_psdrag"
            | "adsb_hybrid_v13_psthrust"
            | "adsb_hybrid_v13b_psthrust_w10"
            | "adsb_hybrid_v13c_psthrust_tet"
            | "adsb_hybrid_v13d_psthrust_parallel"
            | "adsb_hybrid_v13e_psthrust_parallel_l025"
            | "adsb_hybrid_v13f_psthrust_parallel_anneal"
            | "adsb_hybrid_v13g_psthrust_tet_parallel"
            | "adsb_hybrid_v14_psefficiency"
        ):
            # All five hybrid variants share the same X/U/E/DX schema —
            # they only differ in flight_feature_cols (5 / 6 / 6 / 9 / 9),
            # MassEncoder sigmoid temperature (1.0 default vs 3.0 for v5),
            # and the longitudinal head output contract (Newton
            # lift_residual_norm vs CL residual). Schema columns come from
            # the shared module.
            from node_fdm_data.schemas.adsb_hybrid import (
                DX_COLS,
                E0_COLS,
                E1_COLS,
                U_COLS,
                X_COLS,
            )

            arch_name, arch_import = {
                "adsb_hybrid": ("node_adsb_hybrid_v1", "node_fdm.architectures.adsb_hybrid"),
                "adsb_hybrid_v2": (
                    "node_adsb_hybrid_v2",
                    "node_fdm.architectures.adsb_hybrid_v2",
                ),
                "adsb_hybrid_v3": (
                    "node_adsb_hybrid_v3",
                    "node_fdm.architectures.adsb_hybrid_v3",
                ),
                "adsb_hybrid_v4": (
                    "node_adsb_hybrid_v4_mass_features",
                    "node_fdm.architectures.adsb_hybrid_v4",
                ),
                "adsb_hybrid_v5_tempered": (
                    "node_adsb_hybrid_v5_tempered_t3",
                    "node_fdm.architectures.adsb_hybrid_v5_tempered",
                ),
                "adsb_hybrid_v6_mlp": (
                    "node_adsb_hybrid_v6_mlp_monotone",
                    "node_fdm.architectures.adsb_hybrid_v6_mlp",
                ),
                "adsb_hybrid_v7_lean": (
                    "node_adsb_hybrid_v7_lean_2features",
                    "node_fdm.architectures.adsb_hybrid_v7_lean",
                ),
                "adsb_hybrid_v8_lean_t15": (
                    "node_adsb_hybrid_v8_lean_t15",
                    "node_fdm.architectures.adsb_hybrid_v8_lean_t15",
                ),
                "adsb_hybrid_v9_causal": (
                    "node_adsb_hybrid_v9_causal_3features",
                    "node_fdm.architectures.adsb_hybrid_v9_causal",
                ),
                "adsb_hybrid_v10_ps_auxloss": (
                    "node_adsb_hybrid_v10_ps_auxloss",
                    "node_fdm.architectures.adsb_hybrid_v10_ps_auxloss",
                ),
                "adsb_hybrid_v11_ps_residual": (
                    "node_adsb_hybrid_v11_ps_residual",
                    "node_fdm.architectures.adsb_hybrid_v11_ps_residual",
                ),
                "adsb_hybrid_v12_psdrag": (
                    "node_adsb_hybrid_v12_psdrag",
                    "node_fdm.architectures.adsb_hybrid_v12_psdrag",
                ),
                "adsb_hybrid_v13_psthrust": (
                    "node_adsb_hybrid_v13_psthrust",
                    "node_fdm.architectures.adsb_hybrid_v13_psthrust",
                ),
                "adsb_hybrid_v13b_psthrust_w10": (
                    "node_adsb_hybrid_v13b_psthrust_w10",
                    "node_fdm.architectures.adsb_hybrid_v13b_psthrust_w10",
                ),
                "adsb_hybrid_v13c_psthrust_tet": (
                    "node_adsb_hybrid_v13c_psthrust_tet",
                    "node_fdm.architectures.adsb_hybrid_v13c_psthrust_tet",
                ),
                "adsb_hybrid_v13d_psthrust_parallel": (
                    "node_adsb_hybrid_v13d_psthrust_parallel",
                    "node_fdm.architectures.adsb_hybrid_v13d_psthrust_parallel",
                ),
                "adsb_hybrid_v13e_psthrust_parallel_l025": (
                    "node_adsb_hybrid_v13e_psthrust_parallel_l025",
                    "node_fdm.architectures.adsb_hybrid_v13e_psthrust_parallel_l025",
                ),
                "adsb_hybrid_v13f_psthrust_parallel_anneal": (
                    "node_adsb_hybrid_v13f_psthrust_parallel_anneal",
                    "node_fdm.architectures.adsb_hybrid_v13f_psthrust_parallel_anneal",
                ),
                "adsb_hybrid_v13g_psthrust_tet_parallel": (
                    "node_adsb_hybrid_v13g_psthrust_tet_parallel",
                    "node_fdm.architectures.adsb_hybrid_v13g_psthrust_tet_parallel",
                ),
                "adsb_hybrid_v14_psefficiency": (
                    "node_adsb_hybrid_v14_psefficiency",
                    "node_fdm.architectures.adsb_hybrid_v14_psefficiency",
                ),
            }[arch]
            return ArchitectureInfo(
                name=arch_name,
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                e1_cols=E1_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=None,
                segment_filter_fn=None,
                architecture_import=arch_import,
            )
        case _:
            supported = ", ".join(repr(a) for a in _SUPPORTED_ARCHS)
            msg = f"Unknown architecture: {arch!r}. Supported: {supported}."
            raise ValueError(msg)
