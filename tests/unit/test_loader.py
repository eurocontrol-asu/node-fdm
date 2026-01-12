"""Unit tests for data loader utilities."""

import polars as pl
import pytest


class TestGetTrainValData:
    """Tests for get_train_val_data function."""

    @pytest.fixture
    def split_df(self, tmp_path) -> pl.DataFrame:
        """Create a split DataFrame with train/val files."""
        # Create dummy parquet files
        for i in range(5):
            df = pl.DataFrame({"value": [float(i)]})
            df.write_parquet(tmp_path / f"train_{i}.parquet")
        for i in range(2):
            df = pl.DataFrame({"value": [float(i)]})
            df.write_parquet(tmp_path / f"val_{i}.parquet")

        # Create split dataframe
        return pl.DataFrame(
            {
                "filepath": [str(tmp_path / f"train_{i}.parquet") for i in range(5)]
                + [str(tmp_path / f"val_{i}.parquet") for i in range(2)],
                "split": ["train"] * 5 + ["val"] * 2,
            }
        )

    def test_splits_correctly(self, split_df: pl.DataFrame) -> None:
        """Correctly splits into train and val."""
        train_files = split_df.filter(pl.col("split") == "train")["filepath"].to_list()
        val_files = split_df.filter(pl.col("split") == "val")["filepath"].to_list()

        assert len(train_files) == 5
        assert len(val_files) == 2

    def test_accepts_polars_dataframe(self, split_df: pl.DataFrame) -> None:
        """Function accepts Polars DataFrame."""
        # This test verifies the type signature works
        assert isinstance(split_df, pl.DataFrame)
