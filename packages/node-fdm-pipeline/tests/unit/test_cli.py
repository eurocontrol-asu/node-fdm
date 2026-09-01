"""Unit tests for CLI entry points (no I/O)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture


def test_version_cmd(capsys: pytest.CaptureFixture[str]) -> None:
    """version_cmd prints version string."""
    from node_fdm_pipeline.cli import version_cmd

    version_cmd()
    captured = capsys.readouterr()
    assert "node-fdm-pipeline" in captured.out


def test_download_fleet_exits_nonzero_when_any_date_failed(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """A partial campaign must not look successful to a remote supervisor."""
    from node_fdm_pipeline.cli import download_fleet

    cohort = SimpleNamespace(cfg=SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path / "A")))
    plan = SimpleNamespace(cohorts=(cohort,))
    mocker.patch("node_fdm_pipeline.commands._fleet_plan.discover_cohorts", return_value=[])
    mocker.patch("node_fdm_pipeline.commands._fleet_plan.build_fleet_plan", return_value=plan)
    mocker.patch(
        "node_fdm_pipeline.commands._fleet_fetch.download_fleet",
        return_value=[SimpleNamespace(error="boom")],
    )

    with pytest.raises(SystemExit, match="failed on 1 date"):
        download_fleet(
            fleet_dir=tmp_path,
            workers=1,
            data_root=tmp_path,
            dry_run=False,
            force_refresh=False,
        )


def test_download_fleet_rejects_parallelism_before_building_plan(tmp_path: Path) -> None:
    """The CLI reports the measured operating constraint without a traceback."""
    from node_fdm_pipeline.cli import download_fleet

    with pytest.raises(SystemExit, match="workers must be 1"):
        download_fleet(
            fleet_dir=tmp_path,
            workers=2,
            data_root=tmp_path,
            dry_run=True,
            force_refresh=False,
        )
