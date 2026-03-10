"""Pipeline configuration — typed Pydantic models replacing raw YAML access."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, field_validator

__all__ = [
    "BadaConfig",
    "ComputingConfig",
    "PathsConfig",
    "PipelineConfig",
]


class PathsConfig(BaseModel, frozen=True):
    """Directory layout for pipeline data."""

    data_dir: Path
    download_dir: str = "downloaded_parquet"
    preprocess_dir: str = "preprocessed_parquet"
    process_dir: str = "processed_flights"
    predicted_dir: str = "predicted_flights"
    bada_dir: str = "bada_flights"
    models_dir: str = "models"
    figure_dir: str = "figures"
    era5_cache_dir: str = "era5_cache"

    def resolve(self, name: str) -> Path:
        """Resolve a sub-directory path relative to data_dir.

        Args:
            name: Attribute name of the sub-directory (e.g. ``"models_dir"``).

        Returns:
            Absolute path ``data_dir / <sub_dir_value>``.

        Raises:
            AttributeError: If *name* is not a valid attribute.
        """
        value: str = getattr(self, name)
        return self.data_dir / value


class BadaConfig(BaseModel, frozen=True):
    """BADA 4.2 configuration."""

    bada_4_2_dir: Path = Path("TODO")


class ComputingConfig(BaseModel, frozen=True):
    """Computing resources configuration."""

    default_cpu_count: int = 35


class PipelineConfig(BaseModel, frozen=True):
    """Root configuration model — replaces raw YAML dict access.

    Example::

        cfg = PipelineConfig.from_yaml(Path("config.yaml"))
        models = cfg.paths.resolve("models_dir")
    """

    paths: PathsConfig
    typecodes: list[str]
    era5_features: list[str] = []
    computing: ComputingConfig = ComputingConfig()
    bada: BadaConfig = BadaConfig()

    @field_validator("typecodes", mode="before")
    @classmethod
    def _validate_typecodes(cls, v: list[str]) -> list[str]:
        if not v:
            msg = "At least one typecode is required"
            raise ValueError(msg)
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated ``PipelineConfig`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            pydantic.ValidationError: If the YAML content is invalid.
        """
        import yaml

        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)
