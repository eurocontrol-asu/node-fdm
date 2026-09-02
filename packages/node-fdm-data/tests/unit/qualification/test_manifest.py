"""Contrat unitaire du manifeste des preuves de qualification."""

from __future__ import annotations

import pytest

import node_fdm_data


def test_unknown_profile_reports_known_profile_ids() -> None:
    """AC3: un profil inconnu est rejeté en énumérant les identifiants connus."""
    error_type = node_fdm_data.QualificationEvidenceError

    with pytest.raises(error_type) as exc_info:
        node_fdm_data.resolve_evidence_manifest("does-not-exist")

    assert "opensky26-exp03-v1" in str(exc_info.value)
