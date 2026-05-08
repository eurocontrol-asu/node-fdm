from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

from node_fdm_pipeline.commands.train import run_training  # noqa: E402


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
def capture_train_one_typecode(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    from node_fdm_pipeline.commands import train as train_mod

    def fake(ctx, acft):  # type: ignore[no-untyped-def]
        captured["training_config"] = train_mod._build_training_config(ctx, acft)

    def fake_load(_cfg):  # type: ignore[no-untyped-def]
        return None, Path("/tmp")

    monkeypatch.setattr(train_mod, "_train_one_typecode", fake, raising=False)
    monkeypatch.setattr(train_mod, "_load_delta_df", fake_load, raising=False)
    return captured


def test_run_training_passes_cli_override_through_to_trainer(
    tmp_yaml: Path, capture_train_one_typecode: dict
) -> None:
    run_training(arch="qar", config=tmp_yaml, use_mode_weights=True)
    tc = capture_train_one_typecode["training_config"]
    assert tc.use_mode_weights is True

    capture_train_one_typecode.clear()
    run_training(arch="qar", config=tmp_yaml, use_mode_weights=None)
    tc = capture_train_one_typecode["training_config"]
    assert tc.use_mode_weights is False
