"""Hyperparameter sweep matrix for Neural ODE FDM training.

Generates ``RunConfig`` instances for a sweep-by-axis study: one axis varies
at a time around a fixed baseline, replicated across multiple seeds. Used by
``sweep_runner.py`` to orchestrate ``fdm train -> predict -> evaluate`` chains.

Run-time API::

    from scripts.sweep_matrix import default_matrix
    configs = default_matrix(smoke=False)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

__all__ = [
    "Baseline",
    "RunConfig",
    "SweepAxes",
    "build_matrix",
    "default_matrix",
    "run2_matrix",
    "smoke_matrix",
]


class Baseline(BaseModel, frozen=True):
    """Fixed hyperparameter values around which the sweep varies one axis at a time."""

    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3
    use_mode_weights: bool = True
    mode_weight_alpha: float = 0.5
    activation: Literal["silu", "relu", "gelu", "tanh"] = "silu"
    train_limit: int = 5000
    seq_len: int = 60
    method: Literal["euler", "rk4"] = "rk4"
    predict_limit: int | None = None
    backbone_depth: int = 3
    head_depth: int = 2
    hidden_width: int = 48


class SweepAxes(BaseModel, frozen=True):
    """Explicit values to test along each axis (baseline value excluded internally)."""

    batch_sizes: tuple[int, ...] = (32, 64, 128, 256)
    epochs_options: tuple[int, ...] = (50, 100)
    lrs: tuple[float, ...] = (5e-4, 1e-3, 3e-3)
    weightings: tuple[bool, ...] = (True, False)
    alphas: tuple[float, ...] = (0.3, 0.5, 0.7)
    activations: tuple[str, ...] = ("silu", "relu")
    seeds: tuple[int, ...] = (0, 1, 2)


class RunConfig(BaseModel, frozen=True):
    """Concrete hyperparameter combination for a single training run.

    ``run_id`` is derived from the axis and value so that two seeds of the
    same config share the prefix; the file layout (``results/{run_id}/``)
    relies on this for idempotent resume.
    """

    run_id: str
    axis: str
    seed: int

    batch_size: int
    epochs: int
    lr: float
    use_mode_weights: bool
    mode_weight_alpha: float
    activation: str
    train_limit: int
    seq_len: int
    method: str
    predict_limit: int | None = None
    backbone_depth: int = 3
    head_depth: int = 2
    hidden_width: int = 48

    arch: str = "adsb"


def _config_key(
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    use_mode_weights: bool,
    mode_weight_alpha: float,
    activation: str,
) -> tuple[int, int, float, bool, float, str]:
    """Hashable tuple identifying a unique hyperparameter combination.

    ``mode_weight_alpha`` is folded into the key only when weighting is on,
    so disabling weighting collapses all alpha variants to a single config.
    """
    alpha_key = mode_weight_alpha if use_mode_weights else -1.0
    return (batch_size, epochs, lr, use_mode_weights, alpha_key, activation)


def build_matrix(
    *,
    baseline: Baseline,
    axes: SweepAxes,
    arch: str = "adsb",
) -> list[RunConfig]:
    """Build a sweep-by-axis matrix around ``baseline``.

    For each axis, every value not equal to the baseline is emitted as a
    new config. Duplicate hyperparameter tuples across axes are dropped via
    ``_config_key``. Each unique config is replicated for every seed.

    Args:
        baseline: Fixed hyperparameters; appears once as ``baseline_seed{n}``.
        axes: Values to test along each axis.
        arch: Architecture identifier forwarded to ``fdm train --arch``.

    Returns:
        List of :class:`RunConfig`, ordered: baseline first, then by axis.
    """
    seen: set[tuple[int, int, float, bool, float, str]] = set()
    unique: list[tuple[str, dict[str, object]]] = []

    base_kw = {
        "batch_size": baseline.batch_size,
        "epochs": baseline.epochs,
        "lr": baseline.lr,
        "use_mode_weights": baseline.use_mode_weights,
        "mode_weight_alpha": baseline.mode_weight_alpha,
        "activation": baseline.activation,
    }
    unique.append(("baseline", dict(base_kw)))
    seen.add(_config_key(**base_kw))  # type: ignore[arg-type]

    def _try_add(axis: str, label: str, **override: object) -> None:
        kw = {**base_kw, **override}
        key = _config_key(**kw)  # type: ignore[arg-type]
        if key in seen:
            return
        seen.add(key)
        unique.append((f"{axis}_{label}", kw))

    for v in axes.batch_sizes:
        _try_add("bs", str(v), batch_size=v)
    for v in axes.epochs_options:
        _try_add("epochs", str(v), epochs=v)
    for v in axes.lrs:
        _try_add("lr", _format_lr(v), lr=v)
    for v in axes.weightings:
        label = "on" if v else "off"
        _try_add("weighting", label, use_mode_weights=v)
    for v in axes.alphas:
        _try_add("alpha", _format_alpha(v), mode_weight_alpha=v, use_mode_weights=True)
    for v in axes.activations:
        _try_add("activation", v, activation=v)

    runs: list[RunConfig] = []
    for axis_label, kw in unique:
        for seed in axes.seeds:
            run_id = f"{axis_label}_seed{seed}"
            runs.append(
                RunConfig(
                    run_id=run_id,
                    axis=axis_label,
                    seed=seed,
                    train_limit=baseline.train_limit,
                    seq_len=baseline.seq_len,
                    method=baseline.method,
                    predict_limit=baseline.predict_limit,
                    backbone_depth=baseline.backbone_depth,
                    head_depth=baseline.head_depth,
                    hidden_width=baseline.hidden_width,
                    arch=arch,
                    **kw,  # type: ignore[arg-type]
                )
            )
    return runs


def default_matrix(*, arch: str = "adsb") -> list[RunConfig]:
    """Production sweep: full axes, 3 seeds (~30 runs)."""
    return build_matrix(baseline=Baseline(), axes=SweepAxes(), arch=arch)


def run2_matrix(*, arch: str = "adsb") -> list[RunConfig]:
    """Run 2 sweep: focused around run-1 free-wins, exploring lr / seq_len / capacity.

    Baseline (figée d'après run 1) :
        bs=64, lr=1e-3, weighting=off, activation=relu, epochs=50, seq_len=60,
        backbone_depth=3, head_depth=2, hidden_width=48.

    Axes (encadrement symétrique autour de la baseline) :
        * batch_size : {32, 128}
        * lr         : {8.5e-4, 1.5e-3}
        * seq_len    : {30, 120}
        * activation : {gelu}
        * hidden_width   : {24, 96}
        * backbone_depth : {2, 4}

    12 configs x 3 seeds = 36 runs.
    """
    base = Baseline(
        batch_size=64,
        epochs=50,
        lr=1e-3,
        use_mode_weights=False,
        mode_weight_alpha=0.5,
        activation="silu",  # placeholder; baseline run2 forces relu below
        train_limit=5000,
        seq_len=60,
        method="rk4",
        backbone_depth=3,
        head_depth=2,
        hidden_width=48,
    )

    seeds = (0, 1, 2)

    # Each entry: (axis_label, dict of fields overriding baseline for that config)
    variants: list[tuple[str, dict[str, object]]] = [
        ("baseline", {"activation": "relu"}),
        # batch_size
        ("bs_32", {"activation": "relu", "batch_size": 32}),
        ("bs_128", {"activation": "relu", "batch_size": 128}),
        # learning rate
        ("lr_8.5e-4", {"activation": "relu", "lr": 8.5e-4}),
        ("lr_1.5e-3", {"activation": "relu", "lr": 1.5e-3}),
        # sequence length
        ("seq_30", {"activation": "relu", "seq_len": 30}),
        ("seq_120", {"activation": "relu", "seq_len": 120}),
        # activation
        ("act_gelu", {"activation": "gelu"}),
        # hidden width (capacity)
        ("hidden_24", {"activation": "relu", "hidden_width": 24}),
        ("hidden_96", {"activation": "relu", "hidden_width": 96}),
        # backbone depth
        ("backbone_2", {"activation": "relu", "backbone_depth": 2}),
        ("backbone_4", {"activation": "relu", "backbone_depth": 4}),
    ]

    runs: list[RunConfig] = []
    for axis_label, override in variants:
        for seed in seeds:
            kw = {
                "batch_size": base.batch_size,
                "epochs": base.epochs,
                "lr": base.lr,
                "use_mode_weights": base.use_mode_weights,
                "mode_weight_alpha": base.mode_weight_alpha,
                "activation": base.activation,
                "backbone_depth": base.backbone_depth,
                "head_depth": base.head_depth,
                "hidden_width": base.hidden_width,
                "seq_len": base.seq_len,
                **override,
            }
            runs.append(
                RunConfig(
                    run_id=f"{axis_label}_seed{seed}",
                    axis=axis_label,
                    seed=seed,
                    train_limit=base.train_limit,
                    method=base.method,
                    predict_limit=base.predict_limit,
                    arch=arch,
                    **kw,  # type: ignore[arg-type]
                )
            )
    return runs


def smoke_matrix(*, arch: str = "adsb") -> list[RunConfig]:
    """Minimal sweep to verify the train/predict/evaluate pipeline end-to-end.

    Caps:
        * ``train_limit=50`` (vs 5000 in prod)
        * ``epochs=2`` (vs 50)
        * 2 seeds, 4 axes — total 4 unique configs * 2 seeds = 8 runs.

    A failure here surfaces wiring issues (CLI flags, model_name routing,
    parquet parsing) before paying the full training cost.
    """
    smoke_base = Baseline(epochs=2, train_limit=50, batch_size=64, predict_limit=5)
    smoke_axes = SweepAxes(
        batch_sizes=(32, 64),
        epochs_options=(2,),
        lrs=(1e-3,),
        weightings=(True, False),
        alphas=(0.5,),
        activations=("silu", "relu"),
        seeds=(0, 1),
    )
    return build_matrix(baseline=smoke_base, axes=smoke_axes, arch=arch)


def _format_lr(lr: float) -> str:
    """Compact lr label for run_id (e.g. ``5e-4``, ``1e-3``)."""
    return f"{lr:g}"


def _format_alpha(alpha: float) -> str:
    """Compact alpha label (e.g. ``0.3``, ``0.7``)."""
    return f"{alpha:g}"
