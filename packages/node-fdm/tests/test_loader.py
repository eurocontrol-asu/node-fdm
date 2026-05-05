"""Tests for node_fdm.loader — train/val dataset construction from Delta Table data."""

from __future__ import annotations

import numpy as np
import polars as pl

from node_fdm.dataset import FlightDataset
from node_fdm.loader import _fill_nan_sel, _load_and_window, get_train_val_data


def _make_flight_df(
    n_flights: int = 4,
    n_rows: int = 100,
    *,
    train_flights: int = 3,
    include_distance_flag: bool = False,
) -> pl.DataFrame:
    """Create a DataFrame mimicking Delta Table output.

    First ``train_flights`` flights get ``meta_split="train"``,
    the rest get ``meta_split="val"``.
    """
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []

    for i in range(n_flights):
        split = "train" if i < train_flights else "val"
        fid = f"abc123_FLIGHT{i:02d}_s0"
        for j in range(n_rows):
            row: dict[str, object] = {
                "meta_flight_id": fid,
                "meta_split": split,
                "alt": float(np.linspace(1000, 10000, n_rows)[j]),
                "tas": float(np.linspace(200, 250, n_rows)[j]),
                "cmd": float(rng.random()),
                "temp": 220.0,
                "d_alt": float(rng.random()),
            }
            if include_distance_flag:
                row["fdm_flag_distance_ok"] = True
            rows.append(row)

    return pl.DataFrame(rows)


class TestGetTrainValData:
    """Tests for the public get_train_val_data function."""

    def test_basic_train_val(self) -> None:
        """Produces non-empty FlightDataset instances."""
        data_df = _make_flight_df()
        train_ds, val_ds = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
        )
        assert isinstance(train_ds, FlightDataset)
        assert isinstance(val_ds, FlightDataset)
        assert len(train_ds) > 0
        assert len(val_ds) > 0

    def test_sample_shapes(self) -> None:
        """Each sample has correct tensor shapes."""
        data_df = _make_flight_df()
        train_ds, _ = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
        )
        sample = train_ds[0]
        assert sample.x.shape == (10, 2)  # seq_len x n_x
        assert sample.u.shape == (10, 1)
        assert sample.e.shape == (10, 1)
        assert sample.dx.shape == (10, 1)

    def test_train_limit(self) -> None:
        """train_limit restricts number of flights loaded."""
        data_df = _make_flight_df(n_flights=5, train_flights=4)
        ds_full, _ = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
        )
        ds_limited, _ = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
            train_limit=1,
        )
        assert len(ds_limited) < len(ds_full)

    def test_loader_reads_delta_format(self) -> None:
        """Loader accepts DataFrame with meta_flight_id and meta_split columns."""
        data_df = _make_flight_df()
        train_ds, val_ds = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
        )
        assert len(train_ds) > 0
        assert len(val_ds) > 0

    def test_loader_filters_valid(self) -> None:
        """Only rows matching the correct meta_split are loaded per dataset."""
        data_df = _make_flight_df(n_flights=4, train_flights=3)
        train_ds, val_ds = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
        )
        # 3 train flights vs 1 val flight → train should have more samples
        assert len(train_ds) > len(val_ds)

    def test_loader_fills_nan_sel(self) -> None:
        """NaN values in fdm_*_sel columns are replaced by 0.0."""
        data_df = _make_flight_df()
        # Add a _sel column with NaN
        data_df = data_df.with_columns(
            pl.lit(float("nan")).alias("fdm_mach_sel"),
        )
        train_ds, _ = get_train_val_data(
            data_df,
            x_cols=["alt"],
            u_cols=["fdm_mach_sel"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
        )
        # All NaN should have been filled → samples should exist
        assert len(train_ds) > 0
        # Check that u values are 0.0 (filled from NaN)
        sample = train_ds[0]
        assert (sample.u == 0.0).all()

    def test_loader_column_names(self) -> None:
        """Loader works with SI column names from Delta Table."""
        rng = np.random.default_rng(42)
        n = 100
        rows = []
        for i in range(2):
            split = "train" if i == 0 else "val"
            for j in range(n):
                rows.append(
                    {
                        "meta_flight_id": f"flight_{i}",
                        "meta_split": split,
                        # X
                        "raw_alt_m": float(np.linspace(300, 10000, n)[j]),
                        "fdm_gamma_rad": float(rng.uniform(-0.1, 0.1)),
                        "era_tas_ms": float(np.linspace(100, 250, n)[j]),
                        "fdm_heading_rad": float(rng.uniform(0.0, 6.28)),
                        # U
                        "fdm_alt_target_m": float(np.linspace(300, 10000, n)[j]),
                        "fdm_tas_target_ms": float(np.linspace(100, 250, n)[j]),
                        "fdm_gamma_target_rad": float(rng.uniform(-0.1, 0.1)),
                        "fdm_gamma_target_known": True,
                        "fdm_tas_target_known": True,
                        "fdm_heading_target_rad": float(rng.uniform(-3.14, 3.14)),
                        "fdm_heading_target_known": True,
                        "fdm_heading_known": True,
                        # E0
                        "fdm_long_wind_ms": float(rng.uniform(-5, 5)),
                        "era_temp_K": 220.0,
                        "era_u_wind_ms": float(rng.uniform(-10, 10)),
                        "era_v_wind_ms": float(rng.uniform(-10, 10)),
                        # DX
                        "fdm_d_alt_ms": float(rng.uniform(-1, 1)),
                        "fdm_d_gamma_rads": float(rng.uniform(-0.01, 0.01)),
                        "fdm_d_tas_ms2": float(rng.uniform(-1, 1)),
                        "fdm_d_heading_rads": float(rng.uniform(-0.05, 0.05)),
                    }
                )
        data_df = pl.DataFrame(rows)

        from node_fdm_data.schemas.adsb import DX_COLS, E0_COLS, U_COLS, X_COLS

        dx_col_names = [col for _, col in DX_COLS]
        train_ds, val_ds = get_train_val_data(
            data_df,
            x_cols=X_COLS,
            u_cols=U_COLS,
            e_cols=E0_COLS,
            dx_cols=dx_col_names,
            seq_len=10,
            shift=10,
        )
        assert len(train_ds) > 0
        assert len(val_ds) > 0


class TestLoadAndWindow:
    """Tests for _load_and_window edge cases."""

    def _cols(self) -> tuple[list[str], list[str], list[str], list[str]]:
        return ["alt", "tas"], ["cmd"], ["temp"], ["d_alt"]

    def _make_df(self, n: int = 30, flight_id: str = "f0") -> pl.DataFrame:
        """Create a valid flight DataFrame."""
        return pl.DataFrame(
            {
                "meta_flight_id": [flight_id] * n,
                "alt": np.linspace(1000, 10000, n).tolist(),
                "tas": np.linspace(200, 250, n).tolist(),
                "cmd": [0.0] * n,
                "temp": [220.0] * n,
                "d_alt": [1.0] * n,
            }
        )

    def test_missing_columns_skipped(self) -> None:
        """DataFrame with missing columns produces zero samples."""
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f0"] * 20,
                "alt": np.linspace(1000, 10000, 20).tolist(),
                "tas": np.linspace(200, 250, 20).tolist(),
                "cmd": [0.0] * 20,
                "temp": [220.0] * 20,
                # "d_alt" missing
            }
        )
        x, u, e, dx = self._cols()
        samples = _load_and_window(df, x, u, e, dx, seq_len=10, shift=10)
        assert len(samples) == 0

    def test_too_short_flight(self) -> None:
        """Flight shorter than seq_len produces no samples."""
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f0", "f0"],
                "alt": [1000.0, 2000.0],
                "tas": [200.0, 210.0],
                "cmd": [0.0, 0.1],
                "temp": [220.0, 220.0],
                "d_alt": [1.0, 2.0],
            }
        )
        x, u, e, dx = self._cols()
        samples = _load_and_window(df, x, u, e, dx, seq_len=10, shift=10)
        assert len(samples) == 0

    def test_nan_windows_skipped(self) -> None:
        """Windows containing NaN values are excluded."""
        n = 30
        alt = np.linspace(1000, 10000, n)
        alt_nan = alt.copy()
        alt_nan[5] = np.nan

        df_nan = pl.DataFrame(
            {
                "meta_flight_id": ["f0"] * n,
                "alt": alt_nan.tolist(),
                "tas": np.linspace(200, 250, n).tolist(),
                "cmd": [0.0] * n,
                "temp": [220.0] * n,
                "d_alt": [1.0] * n,
            }
        )
        df_clean = pl.DataFrame(
            {
                "meta_flight_id": ["f0"] * n,
                "alt": alt.tolist(),
                "tas": np.linspace(200, 250, n).tolist(),
                "cmd": [0.0] * n,
                "temp": [220.0] * n,
                "d_alt": [1.0] * n,
            }
        )

        x, u, e, dx = self._cols()
        nan_samples = _load_and_window(df_nan, x, u, e, dx, seq_len=10, shift=5)
        clean_samples = _load_and_window(df_clean, x, u, e, dx, seq_len=10, shift=5)
        assert len(nan_samples) < len(clean_samples)

    def test_inf_windows_skipped(self) -> None:
        """Windows containing inf values are excluded."""
        n = 30
        alt = np.linspace(1000, 10000, n)
        alt_inf = alt.copy()
        alt_inf[5] = np.inf

        df_inf = pl.DataFrame(
            {
                "meta_flight_id": ["f0"] * n,
                "alt": alt_inf.tolist(),
                "tas": np.linspace(200, 250, n).tolist(),
                "cmd": [0.0] * n,
                "temp": [220.0] * n,
                "d_alt": [1.0] * n,
            }
        )
        df_clean = pl.DataFrame(
            {
                "meta_flight_id": ["f0"] * n,
                "alt": alt.tolist(),
                "tas": np.linspace(200, 250, n).tolist(),
                "cmd": [0.0] * n,
                "temp": [220.0] * n,
                "d_alt": [1.0] * n,
            }
        )

        x, u, e, dx = self._cols()
        inf_samples = _load_and_window(df_inf, x, u, e, dx, seq_len=10, shift=5)
        clean_samples = _load_and_window(df_clean, x, u, e, dx, seq_len=10, shift=5)
        assert len(inf_samples) < len(clean_samples)

    def test_neg_inf_windows_skipped(self) -> None:
        """Windows containing -inf values are also excluded."""
        n = 20
        tas = np.linspace(200, 250, n)
        tas[15] = -np.inf

        df = pl.DataFrame(
            {
                "meta_flight_id": ["f0"] * n,
                "alt": np.linspace(1000, 10000, n).tolist(),
                "tas": tas.tolist(),
                "cmd": [0.0] * n,
                "temp": [220.0] * n,
                "d_alt": [1.0] * n,
            }
        )

        x, u, e, dx = self._cols()
        samples = _load_and_window(df, x, u, e, dx, seq_len=10, shift=10)
        # Second window (rows 10-19) contains -inf → should be filtered
        assert len(samples) == 1

    def test_distance_flag_filtering(self) -> None:
        """Windows where fdm_flag_distance_ok is False are rejected (AC6)."""
        n = 30
        flags = [True] * n
        flags[5] = False  # One row in first window is bad

        df = pl.DataFrame(
            {
                "meta_flight_id": ["f0"] * n,
                "alt": np.linspace(1000, 10000, n).tolist(),
                "tas": np.linspace(200, 250, n).tolist(),
                "cmd": [0.0] * n,
                "temp": [220.0] * n,
                "d_alt": [1.0] * n,
                "fdm_flag_distance_ok": flags,
            }
        )
        df_clean = df.with_columns(pl.lit(True).alias("fdm_flag_distance_ok"))

        x, u, e, dx = self._cols()
        flagged_samples = _load_and_window(df, x, u, e, dx, seq_len=10, shift=5)
        clean_samples = _load_and_window(df_clean, x, u, e, dx, seq_len=10, shift=5)
        assert len(flagged_samples) < len(clean_samples)

    def test_multiple_flights_grouped(self) -> None:
        """Multiple flights are windowed independently."""
        df1 = self._make_df(n=30, flight_id="f0")
        df2 = self._make_df(n=30, flight_id="f1")
        df = pl.concat([df1, df2])

        x, u, e, dx = self._cols()
        samples = _load_and_window(df, x, u, e, dx, seq_len=10, shift=10)
        # Each flight has 30 rows → 3 windows each → 6 total
        assert len(samples) == 6

    def test_flight_limit(self) -> None:
        """flight_limit restricts the number of flights processed."""
        df1 = self._make_df(n=30, flight_id="f0")
        df2 = self._make_df(n=30, flight_id="f1")
        df = pl.concat([df1, df2])

        x, u, e, dx = self._cols()
        all_samples = _load_and_window(df, x, u, e, dx, seq_len=10, shift=10)
        limited_samples = _load_and_window(df, x, u, e, dx, seq_len=10, shift=10, flight_limit=1)
        assert len(limited_samples) < len(all_samples)


class TestLoaderE1TasDiff:
    """Verify _load_and_window loads fdm_tas_diff_ms into e1 tensor (AXM-771)."""

    def test_loader_e1_tas_diff(self) -> None:
        """DataFrame with fdm_tas_diff_ms column, e1_cols includes it → correct e1 dim."""
        n = 50
        rng = np.random.default_rng(99)
        df = pl.DataFrame(
            {
                "meta_flight_id": ["f0"] * n,
                "alt": np.linspace(1000, 10000, n).tolist(),
                "tas": np.linspace(200, 250, n).tolist(),
                "cmd": [0.0] * n,
                "temp": [220.0] * n,
                "d_alt": [1.0] * n,
                "fdm_tas_diff_ms": rng.uniform(-5, 5, n).tolist(),
            }
        )

        samples = _load_and_window(
            df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
            e1_cols=["fdm_tas_diff_ms"],
        )

        assert len(samples) > 0
        for s in samples:
            assert s.e1 is not None
            assert s.e1.shape == (10, 1)  # seq_len x 1 e1 column
            assert s.e1.isfinite().all()


class TestGetTrainValDataE1Cols:
    """Tests for get_train_val_data with e1_cols (AXM-758)."""

    @staticmethod
    def _make_flight_df_with_e1(
        n_flights: int = 4,
        n_rows: int = 100,
        *,
        train_flights: int = 3,
        e1_col_names: list[str] | None = None,
    ) -> pl.DataFrame:
        """Create a DataFrame with extra E1 columns."""
        rng = np.random.default_rng(42)
        rows: list[dict[str, object]] = []
        e1_col_names = e1_col_names or []

        for i in range(n_flights):
            split = "train" if i < train_flights else "val"
            fid = f"abc123_FLIGHT{i:02d}_s0"
            for j in range(n_rows):
                row: dict[str, object] = {
                    "meta_flight_id": fid,
                    "meta_split": split,
                    "alt": float(np.linspace(1000, 10000, n_rows)[j]),
                    "tas": float(np.linspace(200, 250, n_rows)[j]),
                    "cmd": float(rng.random()),
                    "temp": 220.0,
                    "d_alt": float(rng.random()),
                }
                for col in e1_col_names:
                    row[col] = float(rng.random() * 100)
                rows.append(row)

        return pl.DataFrame(rows)

    def test_loader_e1_cols_stats(self) -> None:
        """E1 columns in DataFrame are loaded and shaped correctly."""
        e1_col_names = ["e1_wind", "e1_temp_delta"]
        data_df = self._make_flight_df_with_e1(e1_col_names=e1_col_names)

        train_ds, val_ds = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            e1_cols=e1_col_names,
            seq_len=10,
            shift=10,
        )

        assert isinstance(train_ds, FlightDataset)
        assert len(train_ds) > 0
        assert len(val_ds) > 0

        # Samples should have e1 tensor with correct shape
        sample = train_ds[0]
        assert hasattr(sample, "e1")
        assert sample.e1 is not None
        assert sample.e1.shape == (10, 2)  # seq_len x n_e1_cols

    def test_loader_no_e1_cols(self) -> None:
        """Calling without e1_cols produces identical behavior — no regression."""
        data_df = _make_flight_df()

        train_ds_baseline, val_ds_baseline = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            seq_len=10,
            shift=10,
        )

        train_ds_explicit, val_ds_explicit = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            e1_cols=None,
            seq_len=10,
            shift=10,
        )

        assert len(train_ds_baseline) == len(train_ds_explicit)
        assert len(val_ds_baseline) == len(val_ds_explicit)

    def test_loader_e1_col_missing_in_dataframe(self) -> None:
        """E1 col not in DataFrame — warning logged, column skipped."""
        data_df = _make_flight_df()  # No E1 columns in DataFrame

        train_ds, val_ds = get_train_val_data(
            data_df,
            x_cols=["alt", "tas"],
            u_cols=["cmd"],
            e_cols=["temp"],
            dx_cols=["d_alt"],
            e1_cols=["nonexistent"],
            seq_len=10,
            shift=10,
        )

        # Should not crash, datasets still valid
        assert len(train_ds) > 0
        assert len(val_ds) > 0

        # e1 should be None or empty since the column was skipped
        sample = train_ds[0]
        if hasattr(sample, "e1") and sample.e1 is not None:
            assert sample.e1.shape[1] == 0  # No valid E1 columns


class TestFillNanSel:
    """Tests for _fill_nan_sel helper."""

    def test_fills_nan_in_sel_columns(self) -> None:
        """NaN in fdm_*_sel columns is filled to 0.0."""
        df = pl.DataFrame(
            {
                "fdm_mach_sel": [0.82, float("nan"), 0.78],
                "fdm_cas_sel_ms": [float("nan"), 150.0, float("nan")],
                "raw_alt_m": [float("nan"), 3000.0, 5000.0],  # Not a _sel col
            }
        )
        result = _fill_nan_sel(df)
        assert result["fdm_mach_sel"].to_list() == [0.82, 0.0, 0.78]
        assert result["fdm_cas_sel_ms"].to_list() == [0.0, 150.0, 0.0]
        # Non-sel column should keep NaN
        assert result["raw_alt_m"].is_nan()[0]

    def test_fills_null_in_sel_columns(self) -> None:
        """Null values in fdm_*_sel columns are also filled to 0.0."""
        df = pl.DataFrame(
            {
                "fdm_vz_sel_ms": [None, 5.0, None],
            }
        )
        result = _fill_nan_sel(df)
        assert result["fdm_vz_sel_ms"].to_list() == [0.0, 5.0, 0.0]
