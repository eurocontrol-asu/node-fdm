"""Tests for TrainingCallback protocol and ConsoleCallback."""

from __future__ import annotations

from node_fdm.callbacks import ConsoleCallback, TrainingCallback


class TestTrainingCallback:
    """Protocol conformance checks."""

    def test_console_callback_is_protocol(self) -> None:
        """ConsoleCallback satisfies TrainingCallback protocol."""
        cb = ConsoleCallback()
        assert isinstance(cb, TrainingCallback)


class TestConsoleCallback:
    """Unit tests for ConsoleCallback."""

    def test_on_epoch_start_no_crash(self) -> None:
        """on_epoch_start runs without error."""
        cb = ConsoleCallback()
        cb.on_epoch_start(epoch=1, total_epochs=10)

    def test_on_epoch_end_no_crash(self) -> None:
        """on_epoch_end runs without error."""
        cb = ConsoleCallback()
        cb.on_epoch_end(
            epoch=1,
            total_epochs=10,
            train_loss=0.5,
            val_loss=0.4,
            is_best=True,
        )

    def test_on_train_end_no_crash(self) -> None:
        """on_train_end runs without error."""
        cb = ConsoleCallback()
        cb.on_train_end(best_val_loss=0.3)
