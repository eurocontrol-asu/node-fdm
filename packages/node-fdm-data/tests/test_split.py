"""Tests for node_fdm_data.split — train/val/test splitting by ICAO24."""

from __future__ import annotations

import polars as pl
import pytest

from node_fdm_data.split import split_by_icao


class TestSplitByIcao:
    """ICAO24-based dataset split tests."""

    @pytest.fixture()
    def delta_df(self) -> pl.DataFrame:
        """Create a DataFrame simulating 100 rows from 10 unique icao24s (10 segments each)."""
        rows = []
        icao24s = [f"abc{i:03d}" for i in range(10)]
        for icao24 in icao24s:
            for seg in range(10):
                rows.append(
                    {
                        "raw_icao24": icao24,
                        "meta_flight_id": f"{icao24}_FL{seg:03d}_s0",
                        "meta_batch_date": "20251108",
                    }
                )
        return pl.DataFrame(rows)

    def test_split_deterministic(self, delta_df: pl.DataFrame) -> None:
        """Same data, 2 runs → identical meta_split."""
        r1 = split_by_icao(delta_df)
        r2 = split_by_icao(delta_df)
        assert r1["meta_split"].to_list() == r2["meta_split"].to_list()

    def test_split_no_leakage(self, delta_df: pl.DataFrame) -> None:
        """All segments of a given icao24 have the same meta_split."""
        result = split_by_icao(delta_df)
        for icao24 in delta_df["raw_icao24"].unique().to_list():
            splits = result.filter(pl.col("raw_icao24") == icao24)["meta_split"].unique()
            assert len(splits) == 1, f"icao24 {icao24} has multiple splits: {splits.to_list()}"

    @pytest.mark.parametrize(
        ("n_icao", "ratios", "expected_mins", "train_max"),
        [
            pytest.param(
                100,
                (0.7, 0.15, 0.15),
                {"train": 60, "val": 5, "test": 5},
                80,
                id="default-70-15-15",
            ),
            pytest.param(
                200,
                (0.5, 0.25, 0.25),
                {"train": 80, "val": 30, "test": 30},
                None,
                id="custom-50-25-25",
            ),
        ],
    )
    def test_split_ratios(
        self,
        n_icao: int,
        ratios: tuple[float, float, float],
        expected_mins: dict[str, int],
        train_max: int | None,
    ) -> None:
        """Hash-based split honours requested ratios within tolerance."""
        rows = [{"raw_icao24": f"icao{i:04d}"} for i in range(n_icao)]
        df = pl.DataFrame(rows)
        result = split_by_icao(df, ratios=ratios)

        counts = result.group_by("meta_split").len()
        split_map = dict(zip(counts["meta_split"].to_list(), counts["len"].to_list(), strict=True))

        for split_name, min_count in expected_mins.items():
            assert split_map.get(split_name, 0) >= min_count
        if train_max is not None:
            assert split_map.get("train", 0) <= train_max

    def test_split_column_added(self, delta_df: pl.DataFrame) -> None:
        """Result has meta_split column with valid values."""
        result = split_by_icao(delta_df)
        assert "meta_split" in result.columns
        assert set(result["meta_split"].unique().to_list()) <= {"train", "val", "test"}

    def test_split_preserves_columns(self, delta_df: pl.DataFrame) -> None:
        """Original columns are preserved."""
        result = split_by_icao(delta_df)
        for col in delta_df.columns:
            assert col in result.columns

    def test_split_empty_df(self) -> None:
        """Empty DataFrame returns empty DataFrame with meta_split."""
        df = pl.DataFrame(
            {"raw_icao24": []},
            schema={"raw_icao24": pl.Utf8},
        )
        result = split_by_icao(df)
        assert len(result) == 0
        assert "meta_split" in result.columns

    def test_split_different_seed(self, delta_df: pl.DataFrame) -> None:
        """Different seeds may produce different assignments."""
        r1 = split_by_icao(delta_df, seed=42)
        r2 = split_by_icao(delta_df, seed=99)
        # Both should be valid
        assert set(r1["meta_split"].unique().to_list()) <= {"train", "val", "test"}
        assert set(r2["meta_split"].unique().to_list()) <= {"train", "val", "test"}
        assert len(r1) == len(r2)
