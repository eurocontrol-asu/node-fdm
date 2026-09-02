from __future__ import annotations


def test_plan_bisection_preserves_exact_ordered_partition_for_one_hundred() -> None:
    """AC1: le lot de 100 est partagé en deux moitiés exactes et ordonnées."""
    from node_fdm_pipeline.commands import _fetch_bisect

    batch = tuple(f"a{i:03d}" for i in range(1, 101))

    plan = _fetch_bisect.plan_bisection(batch, floor=5)

    assert plan.kind == "split"
    assert plan.children == (batch[:50], batch[50:])
    assert plan.children[0] + plan.children[1] == batch
    assert len(set(plan.children[0]) & set(plan.children[1])) == 0


def test_plan_bisection_splits_ten_into_two_batches_of_five() -> None:
    """AC2: le lot de 10 est partagé en deux enfants de cinq appareils."""
    from node_fdm_pipeline.commands import _fetch_bisect

    batch = tuple(f"m{i:03d}" for i in range(1, 11))

    plan = _fetch_bisect.plan_bisection(batch, floor=5)

    assert plan.kind == "split"
    assert plan.children == (batch[:5], batch[5:])


def test_plan_bisection_keeps_batch_at_floor_terminal_and_intact() -> None:
    """AC3: un lot situé au plancher reste intact et ne produit aucun enfant."""
    from node_fdm_pipeline.commands import _fetch_bisect

    batch = tuple(f"b{i:03d}" for i in range(1, 6))

    plan = _fetch_bisect.plan_bisection(batch, floor=5)

    assert plan.kind == "terminal_floor"
    assert plan.batch == batch
    assert plan.children == ()


def test_plan_bisection_rejects_split_below_floor() -> None:
    """AC4: un lot de neuf est terminal car une moitié serait sous le plancher."""
    from node_fdm_pipeline.commands import _fetch_bisect

    batch = tuple(f"n{i:03d}" for i in range(1, 10))

    plan = _fetch_bisect.plan_bisection(batch, floor=5)

    assert plan.kind == "terminal_floor"
    assert plan.batch == batch
    assert plan.children == ()
