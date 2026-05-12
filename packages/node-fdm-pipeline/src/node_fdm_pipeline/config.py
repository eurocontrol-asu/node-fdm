"""Pipeline configuration — typed Pydantic models replacing raw YAML access."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from node_fdm_data.preprocessing.derive import LateralDetectionParams
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "AltFilterConfig",
    "BadaConfig",
    "CasFilterConfig",
    "CleanSpeedsConfig",
    "ComputingConfig",
    "FlagConfig",
    "GammaFilterConfig",
    "LateralDetectionConfig",
    "MachFilterConfig",
    "PathsConfig",
    "PipelineConfig",
    "PreprocessConfig",
    "SelectedParamConfig",
    "TasFilterConfig",
    "TrainingPipelineConfig",
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
    """Mach-number selected-parameter filter.

    Two modes:
    - ``"bilateral_mach"`` (default) — bilateral-smoothed Mach plateau
      detector with altitude-plateau gate (AXM-1689).
    - ``"savgol_mach"`` — legacy savgol detector.
    """

    mode: Literal["bilateral_mach", "savgol_mach"] = "bilateral_mach"
    sigma_s: float = 8.0
    sigma_r: float = 0.08
    n_passes: int = 2
    slope_tol: float = 6.5e-4
    flat_tol: float = 5e-2
    min_len: int = 15
    # Legacy savgol_mach fields kept for backwards compatibility.
    tol: float = 0.0005
    alt_threshold: float = 15000
    smooth_window: int = 30
    use_alt: bool = True


class CasFilterConfig(BaseModel, frozen=True):
    """CAS selected-parameter filter.

    Two modes:
    - ``"bilateral_cas"`` (default) — Butterworth low-pass + bilateral
      smoothing CAS plateau detector (AXM-1689).
    - ``"savgol_cas"`` — legacy savgol detector.
    """

    mode: Literal["bilateral_cas", "savgol_cas"] = "bilateral_cas"
    cutoff_s: float = 180.0
    sigma_s: float = 8.0
    sigma_r: float = 15.0
    n_passes: int = 2
    slope_tol: float = 0.25
    flat_tol: float = 20.0
    min_len: int = 5
    # Legacy savgol_cas fields kept for backwards compatibility.
    tol: float = 0.75
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
    """Vertical-speed selected-parameter filter.

    Two modes:
    - ``"bilateral_vz"`` (default) — bilateral-smoothed vz plateau detector.
    - ``"savgol_vz"`` — legacy savgol detector.
    """

    mode: Literal["bilateral_vz", "savgol_vz"] = "bilateral_vz"
    sigma_s: float = 6.0
    sigma_r: float = 350.0
    slope_tol: float = 15.0
    flat_tol: float = 100.0
    min_len: int = 10
    tol: float = 25
    use_alt: bool = False
    min_abs_value: float = 75
    smooth_window: int = 15
    smooth_method: str = "savgol"


class AltFilterConfig(BaseModel, frozen=True):
    """Altitude selected-parameter filter.

    Two modes:
    - ``"bilateral_vz"`` (default) — detect plateaus where vertical
      speed is locally near zero after a bilateral smoothing of vz.
    - ``"savgol_alt"`` — legacy detector running on a savgol-smoothed
      altitude signal.
    """

    mode: Literal["bilateral_vz", "savgol_alt"] = "bilateral_vz"
    sigma_s: float = 6.0
    sigma_r: float = 350.0
    n_passes: int = 2
    tol_ftmin: float = 150.0
    min_len: int = 6
    tol: float = 25
    use_alt: bool = False
    min_abs_value: float = 25
    smooth_window: int = 5
    smooth_method: str = "savgol"


class GammaFilterConfig(BaseModel, frozen=True):
    """Flight-path angle selected-parameter filter.

    Two modes:
    - ``"bilateral_gamma"`` (default) — bilateral-smoothed gamma plateau detector
      with cascade exclusion (alt-hold mask passed at call site).
    - ``"savgol_gamma"`` — legacy savgol detector.
    """

    mode: Literal["bilateral_gamma", "savgol_gamma"] = "bilateral_gamma"
    sigma_s: float = 6.0
    sigma_r: float = 1.2e-2
    slope_tol: float = 3e-4
    flat_tol: float = 2e-3
    abs_min: float = 5e-3
    min_len: int = 10
    tol: float = 0.002
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
    ``fdm_tas_from_cas_kt`` (TAS recomputed from cleaned IAS via the
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


class LateralDetectionConfig(BaseModel, frozen=True):
    """V3 turn-detector hyperparameters (AXM-1706).

    Forwarded to ``node_fdm_data.lateral.detect_turn_intervals`` and
    ``node_fdm_data.lateral.augment_lateral`` via the ``derive`` stage.
    Defaults match the V3 numerics established by AXM-1704 / AXM-1705 —
    omitting the block keeps behaviour bit-identical.
    """

    bilateral_sigma_s: float = Field(default=8.0, gt=0.0)
    bilateral_sigma_r: float = Field(default=0.01, gt=0.0)
    bilateral_passes: int = Field(default=2, ge=1)
    rate_threshold: float = Field(default=0.05, gt=0.0)

    def to_params(self) -> LateralDetectionParams:
        """Convert to the boundary type consumed by ``derive_columns``."""
        return LateralDetectionParams(
            bilateral_sigma_s=self.bilateral_sigma_s,
            bilateral_sigma_r=self.bilateral_sigma_r,
            bilateral_passes=self.bilateral_passes,
            rate_threshold=self.rate_threshold,
        )


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

    # Mach detection knob (see node_fdm_data.segments). The Mach detector
    # discards plateaus whose mean Mach is below this threshold (low-Mach
    # plateaus during taxi/climb are not cruise-Mach captures).
    mach_min_value: float = 0.5


class TrainingPipelineConfig(BaseModel, frozen=True):
    """Training-time pipeline knobs forwarded to ``node_fdm.trainer.TrainingConfig``.

    Currently exposes the mode-rebalancing toggle (``use_mode_weights``).
    The ``fdm train`` and ``fdm resume`` CLIs accept a runtime override
    flag (``--use-mode-weights / --no-use-mode-weights``); when unset the
    value here is used. Default ``False`` preserves pre-ticket numerics.
    """

    use_mode_weights: bool = False
    mode_weight_alpha: float = 0.5


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
    lateral_detection: LateralDetectionConfig = LateralDetectionConfig()
    training: TrainingPipelineConfig = TrainingPipelineConfig()

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
