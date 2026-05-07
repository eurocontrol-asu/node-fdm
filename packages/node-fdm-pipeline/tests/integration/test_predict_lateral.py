from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skip(
    reason="Requires fixture mini Delta + pretrained lateral checkpoint (not in repo)"
)
def test_predict_writes_lat_lon_for_lateral_arch() -> None:
    pass
