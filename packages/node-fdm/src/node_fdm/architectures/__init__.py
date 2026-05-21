"""Architecture registry and specs (ADS-B v1, QAR)."""

from __future__ import annotations

from node_fdm.architectures import adsb as adsb
from node_fdm.architectures import adsb_hybrid as adsb_hybrid
from node_fdm.architectures import adsb_hybrid_v2 as adsb_hybrid_v2
from node_fdm.architectures import adsb_hybrid_v3 as adsb_hybrid_v3
from node_fdm.architectures import adsb_hybrid_v4 as adsb_hybrid_v4
from node_fdm.architectures import adsb_hybrid_v5_tempered as adsb_hybrid_v5_tempered
from node_fdm.architectures import adsb_hybrid_v6_mlp as adsb_hybrid_v6_mlp
from node_fdm.architectures import adsb_hybrid_v7_lean as adsb_hybrid_v7_lean
from node_fdm.architectures import adsb_hybrid_v8_lean_t15 as adsb_hybrid_v8_lean_t15
from node_fdm.architectures import adsb_hybrid_v9_causal as adsb_hybrid_v9_causal
from node_fdm.architectures import adsb_hybrid_v10_ps_auxloss as adsb_hybrid_v10_ps_auxloss
from node_fdm.architectures import adsb_hybrid_v11_ps_residual as adsb_hybrid_v11_ps_residual
from node_fdm.architectures import adsb_hybrid_v12_psdrag as adsb_hybrid_v12_psdrag
from node_fdm.architectures import qar as qar

__all__ = [
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
    "qar",
    "registry",
]
