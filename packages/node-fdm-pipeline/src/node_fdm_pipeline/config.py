"""Pipeline configuration — typed Pydantic models replacing raw YAML access."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from node_fdm_data.preprocessing.derive import LateralDetectionParams
from pydantic import BaseModel, Field, field_validator, model_validator

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
#
# **The bilateral hyper-parameters are required, on purpose.** They are not
# physical constants: each one is a calibration result, and a different study
# calibrating on different data will land somewhere else. A default here is a
# number nobody cites — it ends up in published results with no trace of where
# it came from, and it silently outlives its own refutation.
#
# That is not hypothetical. These models carried sigma_r = 350 for vz while
# paper_opensky26's Pareto sweep over 1,472 flights retained 100; any config
# omitting the section ran on the superseded value and said nothing about it.
#
# So a config that does not declare them fails to load, naming the channel.
# Writing them down is what makes a result reproducible, and the citation
# belongs in the study's own YAML next to the number:
#
#     selected_params:
#       vz:
#         sigma_r: 100.0      # Poll & Schumann? no — opensky26 tbl.3, Pareto front
#         slope_tol: 50.0
#
# What keeps a default: ``mode`` (which algorithm, not how it is tuned) and the
# legacy savgol fields, which no live config selects and which exist only so an
# old YAML still parses.


class MachFilterConfig(BaseModel, frozen=True):
    """Mach-number selected-parameter filter.

    Two modes:
    - ``"bilateral_mach"`` (default) — bilateral-smoothed Mach plateau
      detector with altitude-plateau gate (AXM-1689).
    - ``"savgol_mach"`` — legacy savgol detector.
    """

    mode: Literal["bilateral_mach", "savgol_mach"] = "bilateral_mach"
    sigma_s: float
    sigma_r: float
    n_passes: int
    slope_tol: float
    flat_tol: float
    min_len: int
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
    cutoff_s: float
    sigma_s: float
    sigma_r: float
    n_passes: int
    slope_tol: float
    flat_tol: float
    min_len: int
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
    sigma_s: float
    sigma_r: float
    slope_tol: float
    flat_tol: float
    min_len: int
    tol: float = 25
    use_alt: bool = False
    min_abs_value: float = 75
    smooth_window: int = 15
    smooth_method: str = "savgol"


class AltFilterConfig(BaseModel, frozen=True):
    """Altitude selected-parameter filter.

    Three modes, and the first two are **different algorithms** rather than
    two tunings of one — they ask different questions of the data:

    - ``"bilateral_alt"`` — a run is a plateau when the **altitude's own
      slope** is flat. Takes ``slope_tol`` and ``flat_tol``; ignores
      ``tol_ftmin``. This is what paper_opensky26 calibrated (sigma_r = 20,
      slope-tol = 6; 60.1% coverage at 99.9% fidelity, its strongest channel),
      so it is the mode to select when reproducing that work.
    - ``"bilateral_vz"`` (default) — a run is a plateau when the **vertical
      speed** is locally near zero. Takes ``tol_ftmin``; ignores
      ``slope_tol``. Never swept, so its tuning is inherited rather than
      measured.
    - ``"savgol_alt"`` — legacy detector running on a savgol-smoothed
      altitude signal.

    ``slope_tol`` and ``flat_tol`` are optional because ``bilateral_vz`` has
    no use for them, and ``tol_ftmin`` likewise for ``bilateral_alt``; the
    validator below requires whichever pair the selected mode consumes. What
    is never optional is the tuning the chosen algorithm actually reads.
    """

    mode: Literal["bilateral_alt", "bilateral_vz", "savgol_alt"] = "bilateral_vz"
    sigma_s: float
    sigma_r: float
    n_passes: int = 2
    tol_ftmin: float | None = None
    slope_tol: float | None = None
    flat_tol: float | None = None
    min_len: int
    tol: float = 25
    use_alt: bool = False
    min_abs_value: float = 25
    smooth_window: int = 5
    smooth_method: str = "savgol"

    @model_validator(mode="after")
    def _require_the_selected_mode_s_tuning(self) -> Self:
        """Demand the knobs the chosen algorithm reads, and only those.

        Optional fields would otherwise let a ``bilateral_alt`` config omit
        ``slope_tol`` and run on nothing — the silent-default failure the
        required fields exist to prevent, reintroduced through a type. Naming
        the mode in the error matters too: the two modes ignore each other's
        knobs, so "slope_tol is missing" alone would read as a typo rather
        than as a choice of detector.
        """
        needed = {
            "bilateral_alt": ("slope_tol", "flat_tol"),
            "bilateral_vz": ("tol_ftmin",),
            "savgol_alt": (),
        }[self.mode]
        missing = [name for name in needed if getattr(self, name) is None]
        if missing:
            msg = (
                f"alt mode {self.mode!r} reads {', '.join(needed)}; "
                f"missing {', '.join(missing)}. These are calibration results, "
                f"so the config has to state them."
            )
            raise ValueError(msg)
        return self


class GammaFilterConfig(BaseModel, frozen=True):
    """Flight-path angle selected-parameter filter.

    Two modes:
    - ``"bilateral_gamma"`` (default) — bilateral-smoothed gamma plateau detector
      with cascade exclusion (alt-hold mask passed at call site).
    - ``"savgol_gamma"`` — legacy savgol detector.
    """

    mode: Literal["bilateral_gamma", "savgol_gamma"] = "bilateral_gamma"
    sigma_s: float
    sigma_r: float
    slope_tol: float
    flat_tol: float
    abs_min: float
    min_len: int
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

    Groups all per-parameter filter thresholds.

    The five detector channels are **required**: each carries calibrated
    hyper-parameters, and a study that does not state them cannot be
    reproduced from its own config. ``tas`` keeps a default because it has no
    bilateral detector — it runs the legacy savgol path, whose thresholds are
    tolerances rather than a calibration result.

    .. note:: Legacy v1 values for reference:
       mach.tol=0.002, cas.tol=1.0, vz.min_abs_value=50.
    """

    mach: MachFilterConfig
    cas: CasFilterConfig
    tas: TasFilterConfig = TasFilterConfig()
    vz: VzFilterConfig
    alt: AltFilterConfig
    gamma: GammaFilterConfig

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
    # Required: see the note above the filter models. A config silent about
    # its detector tuning would run on whatever this file last happened to
    # hold, and publish results nothing traces back.
    selected_params: SelectedParamConfig
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
    def from_yaml(cls, path: Path, *, data_root: Path | None = None) -> Self:
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated ``PipelineConfig`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            pydantic.ValidationError: If the YAML content is invalid.
        """
        import os

        import yaml

        data = yaml.safe_load(path.read_text())
        cfg = cls.model_validate(data)

        configured_root = data_root
        if configured_root is None and (environment_root := os.environ.get("NODE_FDM_DATA")):
            configured_root = Path(environment_root).expanduser()
        if configured_root is None:
            return cfg

        root = configured_root.expanduser().resolve()
        cohort_name = cfg.paths.data_dir.name
        paths = cfg.paths.model_copy(
            update={
                "data_dir": root / cohort_name,
                "era5_cache_dir": str(root / "era5_cache"),
            }
        )
        return cfg.model_copy(update={"paths": paths})
