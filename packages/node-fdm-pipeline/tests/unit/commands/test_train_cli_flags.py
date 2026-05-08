from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from node_fdm_pipeline.cli import app


def _run(args: list[str]) -> None:
    """Invoke cyclopts app and swallow the success-path SystemExit(0)."""
    with contextlib.suppress(SystemExit):
        app(args)


@pytest.fixture
def tmp_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        "paths:\n  data_dir: /tmp/data\n"
        "typecodes: [A320]\n"
        "bada:\n  bada_4_2_dir: /tmp/bada\n"
        "training:\n  use_mode_weights: false\n"
    )
    return p


@pytest.fixture
def capture_run_training(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def fake_run_training(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "node_fdm_pipeline.commands.train.run_training",
        fake_run_training,
        raising=False,
    )
    return captured


def test_train_cli_accepts_use_mode_weights_flag(
    tmp_yaml: Path, capture_run_training: dict
) -> None:
    _run(["train", "--arch", "qar", "--config", str(tmp_yaml), "--use-mode-weights"])
    assert capture_run_training.get("use_mode_weights") is True


def test_train_cli_accepts_no_use_mode_weights_flag(
    tmp_yaml: Path, capture_run_training: dict
) -> None:
    _run(
        [
            "train",
            "--arch",
            "qar",
            "--config",
            str(tmp_yaml),
            "--no-use-mode-weights",
        ]
    )
    assert capture_run_training.get("use_mode_weights") is False


def test_train_cli_unset_flag_yields_none_override(
    tmp_yaml: Path, capture_run_training: dict
) -> None:
    _run(["train", "--arch", "qar", "--config", str(tmp_yaml)])
    assert capture_run_training.get("use_mode_weights") is None


@pytest.fixture
def capture_run_resume(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def fake_run_resume(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "node_fdm_pipeline.commands.resume.run_resume",
        fake_run_resume,
        raising=False,
    )
    return captured


def test_resume_cli_accepts_use_mode_weights_flag(
    tmp_path: Path, tmp_yaml: Path, capture_run_resume: dict
) -> None:
    model = tmp_path / "model.pt"
    model.mkdir()
    _run(
        [
            "resume",
            "--model",
            str(model),
            "--config",
            str(tmp_yaml),
            "--use-mode-weights",
        ]
    )
    assert capture_run_resume.get("use_mode_weights") is True
