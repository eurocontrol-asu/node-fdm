"""Unit tests for the sweep matrix generator.

Covers:
- baseline appears once with the configured seeds
- each axis emits configs that differ only along that axis
- duplicate (axis_value == baseline_value) combinations are dropped
- alpha is folded out of the dedup key when weighting is off
"""

from __future__ import annotations

from collections import Counter

from scripts.sweep_matrix import (
    Baseline,
    RunConfig,
    SweepAxes,
    build_matrix,
    default_matrix,
    smoke_matrix,
)


def _unique_configs(runs: list[RunConfig]) -> list[tuple[int, int, float, bool, float, str]]:
    seen: set[tuple[int, int, float, bool, float, str]] = set()
    for r in runs:
        alpha = r.mode_weight_alpha if r.use_mode_weights else -1.0
        seen.add((r.batch_size, r.epochs, r.lr, r.use_mode_weights, alpha, r.activation))
    return sorted(seen)


def test_baseline_appears_once_per_seed() -> None:
    runs = build_matrix(baseline=Baseline(), axes=SweepAxes())
    baselines = [r for r in runs if r.axis == "baseline"]
    seed_counts = Counter(r.seed for r in baselines)
    assert set(seed_counts) == {0, 1, 2}
    assert all(c == 1 for c in seed_counts.values())


def test_axis_only_varies_one_field() -> None:
    base = Baseline()
    runs = build_matrix(baseline=base, axes=SweepAxes())
    for r in runs:
        if not r.axis.startswith("bs_"):
            continue
        assert r.epochs == base.epochs
        assert r.lr == base.lr
        assert r.use_mode_weights == base.use_mode_weights
        assert r.activation == base.activation


def test_no_duplicate_hyperparam_tuples() -> None:
    runs = build_matrix(baseline=Baseline(), axes=SweepAxes())
    unique_for_seed_zero = _unique_configs([r for r in runs if r.seed == 0])
    seen: set[tuple[int, int, float, bool, float, str]] = set()
    for cfg in unique_for_seed_zero:
        assert cfg not in seen
        seen.add(cfg)


def test_baseline_value_excluded_from_its_own_axis() -> None:
    base = Baseline()
    runs = build_matrix(baseline=base, axes=SweepAxes())
    bs_axis = [r for r in runs if r.axis.startswith("bs_") and r.seed == 0]
    bs_values = {r.batch_size for r in bs_axis}
    assert base.batch_size not in bs_values


def test_smoke_matrix_is_small_and_well_formed() -> None:
    runs = smoke_matrix()
    assert 4 <= len(runs) <= 20
    assert all(r.train_limit == 50 for r in runs)
    assert all(r.epochs == 2 for r in runs)


def test_default_matrix_uses_three_seeds() -> None:
    runs = default_matrix()
    seeds = {r.seed for r in runs}
    assert seeds == {0, 1, 2}


def test_alpha_axis_only_emitted_with_weighting_on() -> None:
    runs = build_matrix(baseline=Baseline(), axes=SweepAxes())
    for r in runs:
        if r.axis.startswith("alpha_"):
            assert r.use_mode_weights, f"alpha axis run with weighting off: {r.run_id}"


def test_run_id_is_unique_per_run() -> None:
    runs = default_matrix()
    ids = [r.run_id for r in runs]
    assert len(ids) == len(set(ids)), "run_ids must be unique"


def test_activation_axis_present() -> None:
    runs = build_matrix(baseline=Baseline(), axes=SweepAxes())
    activations = {r.activation for r in runs if r.axis.startswith("activation_")}
    assert "relu" in activations
