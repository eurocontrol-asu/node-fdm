"""Training callbacks for the ODE trainer.

Provides a :class:`TrainingCallback` protocol and a default
:class:`ConsoleCallback` that logs epoch metrics via ``structlog``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog

__all__ = [
    "ConsoleCallback",
    "TrainingCallback",
]


@runtime_checkable
class TrainingCallback(Protocol):
    """Protocol for training event hooks."""

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Called at the beginning of each epoch.

        Args:
            epoch: Current epoch number (1-based).
            total_epochs: Total number of epochs.
        """
        ...

    def on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        *,
        is_best: bool = False,
        lr: float | None = None,
    ) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number (1-based).
            total_epochs: Total number of epochs.
            train_loss: Average training loss for the epoch.
            val_loss: Average validation loss for the epoch.
            is_best: Whether this epoch achieved a new best validation loss.
            lr: Current learning rate after any scheduler step.
        """
        ...

    def on_train_end(self, best_val_loss: float) -> None:
        """Called after training completes.

        Args:
            best_val_loss: Best validation loss achieved during training.
        """
        ...


class ConsoleCallback:
    """Default callback that logs training progress via ``structlog``.

    Implements :class:`TrainingCallback` using structured logging instead
    of ``print()`` statements.
    """

    def __init__(self) -> None:
        self._log = structlog.get_logger("node_fdm.training")

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log epoch start."""
        self._log.debug("epoch_start", epoch=epoch, total_epochs=total_epochs)

    def on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        *,
        is_best: bool = False,
        lr: float | None = None,
    ) -> None:
        """Log epoch metrics."""
        self._log.info(
            "epoch_complete",
            epoch=epoch,
            total_epochs=total_epochs,
            train_loss=round(train_loss, 6),
            val_loss=round(val_loss, 6),
            is_best=is_best,
            lr=lr,
        )

    def on_train_end(self, best_val_loss: float) -> None:
        """Log training completion."""
        self._log.info("training_complete", best_val_loss=round(best_val_loss, 6))
