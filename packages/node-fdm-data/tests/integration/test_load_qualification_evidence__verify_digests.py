"""Contrats d'intégration des preuves de qualification embarquées."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

import node_fdm_data

pytestmark = pytest.mark.integration

_PROFILE_ID = "opensky26-exp03-v1"
_SOURCE_COMMIT = "80e8039f85a59dc214630386ca637ea7a296cfa3"
_EXPECTED_DIGESTS = {
    "retained.csv": "2654c9b688669722271246868633dca0598cc345651e54feb1245affa35b58e1",
    "grids.json": "55a80354047a8d5c61f41921b4541f98699e9a158db5c427a99a9d6f5c288805",
    "metrics.yaml": "3c14d9a77705511296121f63bf2d957679b200d23e63b150a5f0d8ffc54a0a14",
}


def test_packaged_evidence_matches_frozen_provenance_and_digests() -> None:
    """AC1: les octets embarqués correspondent au commit et aux empreintes figés."""
    bundle = node_fdm_data.load_qualification_evidence(_PROFILE_ID)

    recomputed = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.evidence_root.iterdir()
    }

    assert bundle.provenance.source_commit == _SOURCE_COMMIT
    assert dict(bundle.digests) == _EXPECTED_DIGESTS
    assert recomputed == _EXPECTED_DIGESTS


def test_altered_retained_csv_fails_closed_with_both_digests(tmp_path: Path) -> None:
    """AC2: un octet altéré ferme le chargement et expose les deux empreintes."""
    bundle = node_fdm_data.load_qualification_evidence(_PROFILE_ID)
    copied_evidence = tmp_path / "evidence"
    shutil.copytree(bundle.evidence_root, copied_evidence)
    retained = copied_evidence / "retained.csv"
    retained.write_bytes(retained.read_bytes() + b"\x00")
    observed_digest = hashlib.sha256(retained.read_bytes()).hexdigest()
    expected_digest = _EXPECTED_DIGESTS["retained.csv"]
    error_type = node_fdm_data.QualificationEvidenceError

    with pytest.raises(error_type) as exc_info:
        node_fdm_data.verify_digests(copied_evidence, _EXPECTED_DIGESTS)

    message = str(exc_info.value)
    assert "retained.csv" in message
    assert expected_digest in message
    assert observed_digest in message


def test_evidence_root_is_inside_package_and_contains_only_evidence_files() -> None:
    """AC4: la racine embarquée est interne au paquet et contient trois fichiers."""
    package_root = Path(node_fdm_data.__file__).parent.resolve()
    bundle = node_fdm_data.load_qualification_evidence(_PROFILE_ID)
    evidence_root = bundle.evidence_root.resolve()

    evidence_root.relative_to(package_root)
    assert evidence_root.is_dir()
    assert sorted(path.name for path in evidence_root.iterdir()) == [
        "grids.json",
        "metrics.yaml",
        "retained.csv",
    ]
