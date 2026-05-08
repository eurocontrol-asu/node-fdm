from __future__ import annotations

import torch

from node_fdm.dataset import FlightSample
from node_fdm.trainer import _collate_flight_samples

SEQ_LEN = 60


def _mk_sample(*, with_w: bool) -> FlightSample:
    kwargs = {
        "x": torch.zeros(SEQ_LEN, 3),
        "u": torch.zeros(SEQ_LEN, 5),
        "e": torch.zeros(SEQ_LEN, 2),
        "dx": torch.zeros(SEQ_LEN, 3),
    }
    if with_w:
        kwargs["w"] = torch.full((SEQ_LEN,), 1.0)
    return FlightSample(**kwargs)


def test_collate_returns_w_when_all_samples_have_w() -> None:
    batch = [_mk_sample(with_w=True) for _ in range(4)]
    out = _collate_flight_samples(batch)
    assert out[-1].shape == (4, SEQ_LEN)


def test_collate_omits_w_when_no_sample_has_w() -> None:
    batch = [_mk_sample(with_w=False) for _ in range(4)]
    out = _collate_flight_samples(batch)
    assert len(out) == 4
