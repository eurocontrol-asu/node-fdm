"""Coverage tests for thin branches in commands/data.py."""

from __future__ import annotations

import builtins
from pathlib import Path

import polars as pl
import pytest

from _config_fixtures import SELECTED_PARAMS_YAML


class TestAircraftListMissingTraffic:
    def test_aircraft_list_exits_when_traffic_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """aircraft_list non-dry-run with `traffic` unavailable → SystemExit(1)."""
        from node_fdm_pipeline.commands.data import aircraft_list

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(
            f'paths:\n  data_dir: "{data_dir}"\n\ntypecodes:\n  - A320\n' + SELECTED_PARAMS_YAML
        )

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "traffic":
                raise ImportError("simulated missing traffic")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(SystemExit) as exc:
            aircraft_list(config=config, dry_run=False)
        assert exc.value.code == 1


class TestSplitCommand:
    def test_split_writes_meta_split(self, tmp_path: Path) -> None:
        """split() with non-dry-run adds meta_split column to the Delta table."""
        from node_fdm_pipeline.commands.data import split

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        delta_table = data_dir / "flights.delta"

        df = pl.DataFrame(
            {
                "raw_icao24": [f"icao{i:04d}" for i in range(20) for _ in range(3)],
                "raw_timestamp": list(range(60)),
            }
        )
        df.write_delta(str(delta_table), mode="overwrite")

        config = tmp_path / "config.yaml"
        config.write_text(
            f'paths:\n  data_dir: "{data_dir}"\n\ntypecodes:\n  - A320\n' + SELECTED_PARAMS_YAML
        )

        split(config=config, ratios=(0.7, 0.15, 0.15), seed=42)

        out = pl.read_delta(str(delta_table))
        assert "meta_split" in out.columns
        assert set(out["meta_split"].unique().to_list()).issubset({"train", "val", "test"})
