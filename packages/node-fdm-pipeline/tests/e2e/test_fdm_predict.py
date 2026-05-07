from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.skip(reason="Requires fixture model + config; covered by unit + integration tiers")
def test_predict_lateral_arch_includes_lat_lon() -> None:
    pass
