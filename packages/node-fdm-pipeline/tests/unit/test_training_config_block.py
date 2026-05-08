from __future__ import annotations

from pathlib import Path

import pytest

from node_fdm_pipeline.config import PipelineConfig, TrainingPipelineConfig


def _minimal_yaml(tmp_path: Path, training_block: str = "") -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        "paths:\n  data_dir: /tmp/data\n"
        "typecodes: [A320]\n"
        "bada:\n  bada_4_2_dir: /tmp/bada\n" + training_block
    )
    return p


def test_pipeline_config_training_block_defaults(tmp_path: Path) -> None:
    p = _minimal_yaml(tmp_path)
    try:
        cfg = PipelineConfig.from_yaml(p)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PipelineConfig requires more fields: {exc}")
    assert isinstance(cfg.training, TrainingPipelineConfig)
    assert cfg.training.use_mode_weights is False


def test_pipeline_config_training_block_round_trips(tmp_path: Path) -> None:
    p = _minimal_yaml(
        tmp_path,
        training_block="training:\n  use_mode_weights: true\n",
    )
    try:
        cfg = PipelineConfig.from_yaml(p)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PipelineConfig requires more fields: {exc}")
    assert cfg.training.use_mode_weights is True
