"""Pipeline configuration — typed Pydantic models replacing raw YAML access."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, field_validator

__all__ = [
    "AltFilterConfig",
    "BadaConfig",
    "CasFilterConfig",
    "CleanSpeedsConfig",
    "ComputingConfig",
    "FlagConfig",
    "GammaFilterConfig",
    "MachFilterConfig",
    "PathsConfig",
    "PipelineConfig",
    "PreprocessConfig",
    "SelectedParamConfig",
    "TasFilterConfig",
    "VzFilterConfig",
]


class PathsConfig(BaseModel, frozen=True):
    """Directory layout for pipeline data."""

    data_dir: Path
    preprocess_dir: str = "preprocessed_parquet"
    process_dir: str = "processed_flights"
    predicted_dir: str = "predicted_flights"
    bada_dir: str = "bada_flights"
    models_dir: str = "models"
    figure_dir: str = "figures"
    era5_cache_dir: str = "era5_cache"
    delta_table: str = "flights.delta"

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


# ---------------------------------------------------------------------------
# Selected-parameter filter configs (externalized from opensky_v2)
# ---------------------------------------------------------------------------


class MachFilterConfig(BaseModel, frozen=True):
    """Mach-number selected-parameter filter."""

    tol: float = 0.0005
    min_len: int = 120
    alt_threshold: float = 15000
    smooth_window: int = 30
    use_alt: bool = True


class CasFilterConfig(BaseModel, frozen=True):
    """CAS selected-parameter filter."""

    tol: float = 0.75
    min_len: int = 20
    use_alt: bool = False
    smooth_window: int = 20
    smooth_method: str = "savgol"


class TasFilterConfig(BaseModel, frozen=True):
    """TAS selected-parameter filter."""

    tol: float = 0.75
    min_len: int = 20
    use_alt: bool = False
    smooth_window: int = 20
    smooth_method: str = "savgol"


class VzFilterConfig(BaseModel, frozen=True):
    """Vertical-speed selected-parameter filter."""

    tol: float = 25
    min_len: int = 25
    use_alt: bool = False
    min_abs_value: float = 75
    smooth_window: int = 15
    smooth_method: str = "savgol"


class AltFilterConfig(BaseModel, frozen=True):
    """Altitude selected-parameter filter."""

    tol: float = 25
    min_len: int = 5
    use_alt: bool = False
    min_abs_value: float = 25
    smooth_window: int = 5
    smooth_method: str = "savgol"


class GammaFilterConfig(BaseModel, frozen=True):
    """Flight-path angle selected-parameter filter.

    Adds ``min_abs_value`` (default 0.005 rad) to filter near-zero
    gamma plateaus during cruise that are not meaningful targets.
    """

    tol: float = 0.002
    min_len: int = 15
    use_alt: bool = False
    min_abs_value: float = 0.005
    smooth_window: int = 5
    smooth_method: str = "savgol"


class PreprocessConfig(BaseModel, frozen=True):
    """Resampling and interpolation configuration for étape 1.5."""

    rate_s: int = 4
    max_gap_s: float = 30.0
    min_duration_s: int = 240
    smooth: bool = True


class CleanSpeedsConfig(BaseModel, frozen=True):
    """Cleaning of BDS (Mode-S) and ERA5 speed signals.

    Applied per flight by the ``clean-speeds`` stage between ``enrich``
    and ``derive``.  Produces ``bds_*_clean`` columns plus a derived
    ``bds_tas_from_cas_kt`` (TAS recomputed from cleaned IAS via the
    ERA5 static temperature).

    Defaults are calibrated against the visual validation in
    ``scripts/check_speed_sources_v3.py`` on the production dataset.
    """

    bds_window: int = 50
    era_window: int = 15
    k: float = 3.0
    n_passes: int = 3
    interp_max_gap: int = 10
    frozen_min_run_len_mach: int = 20
    frozen_min_run_len_ias: int = 20
    frozen_min_run_len_tas: int = 6
    point_jump_max_mach: float = 0.05
    point_jump_max_kt: float = 20.0
    zigzag_jump_min_mach: float = 0.05
    zigzag_jump_min_kt: float = 20.0
    zigzag_half_window: int = 15
    zigzag_density_min_bds: float = 0.25
    zigzag_density_min_era: float = 0.15
    on_ground_vz_threshold: float = 200.0
    on_ground_alt_threshold: float = 1500.0


class FlagConfig(BaseModel, frozen=True):
    """Validity flag thresholds for pipeline v3 étape 2."""

    min_points: int = 40
    min_speed_kt: float = 90.0
    distance_low_thr: float = 200.0
    distance_upper_thr: float = 3000.0


class SelectedParamConfig(BaseModel, frozen=True):
    """Selected-parameter filter configuration.

    Groups all per-parameter filter thresholds.  Defaults match the
    ``opensky_v2`` branch values.

    .. note:: Legacy v1 values for reference:
       mach.tol=0.002, cas.tol=1.0, vz.min_abs_value=50.
    """

    mach: MachFilterConfig = MachFilterConfig()
    cas: CasFilterConfig = CasFilterConfig()
    tas: TasFilterConfig = TasFilterConfig()
    vz: VzFilterConfig = VzFilterConfig()
    alt: AltFilterConfig = AltFilterConfig()
    gamma: GammaFilterConfig = GammaFilterConfig()

    # Crossover-aware Mach/CAS detection knobs (see node_fdm_data.segments)
    cas_deviation_kt: float = 5.0
    mach_min_value: float = 0.5
    transition_margin: int = 30
    cas_search_window: int = 60


class PipelineConfig(BaseModel, frozen=True):
    """Root configuration model — replaces raw YAML dict access.

    Example::

        cfg = PipelineConfig.from_yaml(Path("config.yaml"))
        models = cfg.paths.resolve("models_dir")
    """

    paths: PathsConfig
    typecodes: list[str]
    era5_features: list[str] = []
    era5_null_threshold: float = 0.05
    computing: ComputingConfig = ComputingConfig()
    bada: BadaConfig = BadaConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    flag: FlagConfig = FlagConfig()
    selected_params: SelectedParamConfig = SelectedParamConfig()
    clean_speeds: CleanSpeedsConfig = CleanSpeedsConfig()

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
