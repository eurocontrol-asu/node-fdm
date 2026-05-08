from __future__ import annotations

import torch

from node_fdm.dataset import FlightSample


def _mk(seq_len: int = 60) -> dict[str, torch.Tensor]:
    return {
        "x": torch.zeros(seq_len, 3),
        "u": torch.zeros(seq_len, 5),
        "e": torch.zeros(seq_len, 2),
        "dx": torch.zeros(seq_len, 3),
    }


def test_flight_sample_optional_w_field_defaults_to_none() -> None:
    sample = FlightSample(**_mk())
    assert sample.w is None


def test_flight_sample_carries_w_tensor_when_provided() -> None:
    sample = FlightSample(**_mk(), w=torch.ones(60))
    assert sample.w is not None
    assert sample.w.shape == (60,)
    assert sample.w.dtype == torch.float32
