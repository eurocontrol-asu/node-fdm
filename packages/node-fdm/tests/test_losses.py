"""Tests for loss factory."""

from __future__ import annotations

import pytest
import torch.nn as nn

from node_fdm.losses import get_loss


class TestGetLoss:
    """Unit tests for get_loss factory."""

    def test_mse(self) -> None:
        """Returns MSELoss instance."""
        loss = get_loss("mse")
        assert isinstance(loss, nn.MSELoss)

    def test_bce(self) -> None:
        """Returns BCELoss instance."""
        loss = get_loss("bce")
        assert isinstance(loss, nn.BCELoss)

    def test_huber(self) -> None:
        """Returns HuberLoss instance."""
        loss = get_loss("huber")
        assert isinstance(loss, nn.HuberLoss)

    def test_l1(self) -> None:
        """Returns L1Loss instance."""
        loss = get_loss("l1")
        assert isinstance(loss, nn.L1Loss)

    def test_case_insensitive(self) -> None:
        """Case is normalised to lowercase."""
        loss = get_loss("MSE")
        assert isinstance(loss, nn.MSELoss)

    def test_unknown_raises(self) -> None:
        """Unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss("unknown_loss_xyz")
