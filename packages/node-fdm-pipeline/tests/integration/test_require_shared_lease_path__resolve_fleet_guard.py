from __future__ import annotations

from pathlib import Path

import pytest

from node_fdm_pipeline.commands import _fleet_boundary as boundary
from node_fdm_pipeline.commands._trino_lease import LeaseConfigError

pytestmark = pytest.mark.integration

_RESOLVE_GUARD_SYMBOL = "resolve_fleet_guard"


def test_resolve_fleet_guard_rejects_missing_lease_parent_without_writes(
    tmp_path: Path,
) -> None:
    """AC5: an invalid lease parent propagates LeaseConfigError without writes."""
    selection = tmp_path / "selection.json"
    resolved_config = tmp_path / "resolved-config.json"
    profile = tmp_path / "profile.json"
    selection.write_text('{"selection":"v1"}', encoding="utf-8")
    resolved_config.write_text('{"workers":2}', encoding="utf-8")
    profile.write_text('{"aircraft":"A320"}', encoding="utf-8")
    missing_parent = tmp_path / "absent"
    lease_path = missing_parent / "lease.json"
    resolve_fleet_guard = getattr(boundary, _RESOLVE_GUARD_SYMBOL)

    with pytest.raises(LeaseConfigError):
        resolve_fleet_guard(
            selection=selection,
            resolved_config=resolved_config,
            profile=profile,
            lease_path=lease_path,
        )

    assert not missing_parent.exists()
