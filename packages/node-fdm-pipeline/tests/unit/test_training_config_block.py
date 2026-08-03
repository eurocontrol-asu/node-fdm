from __future__ import annotations

from pathlib import Path

from _config_fixtures import SELECTED_PARAMS_YAML
from node_fdm_pipeline.config import PipelineConfig, TrainingPipelineConfig


def _minimal_yaml(tmp_path: Path, training_block: str = "") -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(
        "paths:\n  data_dir: /tmp/data\n"
        "typecodes: [A320]\n"
        "bada:\n  bada_4_2_dir: /tmp/bada\n" + SELECTED_PARAMS_YAML + training_block
    )
    return p


# The try/except pytest.skip that used to wrap each load is gone: it swallowed
# every exception, so making the detector params required silently disabled
# both tests instead of failing them. The config is now complete, so the loads
# are expected to succeed and any error is a real failure.


def test_pipeline_config_training_block_defaults(tmp_path: Path) -> None:
    cfg = PipelineConfig.from_yaml(_minimal_yaml(tmp_path))
    assert isinstance(cfg.training, TrainingPipelineConfig)
    assert cfg.training.use_mode_weights is False


def test_pipeline_config_training_block_round_trips(tmp_path: Path) -> None:
    p = _minimal_yaml(
        tmp_path,
        training_block="training:\n  use_mode_weights: true\n",
    )
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.training.use_mode_weights is True
