"""Tests for the official architecture provider catalog."""

from __future__ import annotations

from node_fdm.architectures import get
from node_fdm_models import NODE_ADSB_V1, QAR, architectures


def test_catalog_exposes_stable_aliases() -> None:
    catalog = architectures()
    assert catalog["adsb"] is NODE_ADSB_V1
    assert catalog[NODE_ADSB_V1.name] is NODE_ADSB_V1
    assert catalog[QAR.name] is QAR


def test_installed_provider_is_discovered_without_explicit_import() -> None:
    assert get("adsb").name == "node_adsb_v1"
    assert get("qar").name == "qar"


def test_official_specs_reference_runtime_layers_only() -> None:
    for spec in (NODE_ADSB_V1, QAR):
        assert spec.layers
        assert all(layer.layer_class.startswith("node_fdm.layers.") for layer in spec.layers)
