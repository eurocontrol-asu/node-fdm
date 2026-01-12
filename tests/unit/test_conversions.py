"""Unit tests for data conversion utilities."""

import numpy as np

from node_fdm.utils.data.conversions import (
    AdditionUnitConverter,
    CategoryMapper,
    LinearUnitConverter,
    correct_float,
    correct_str,
    identity,
)


class TestCorrectFloat:
    """Tests for correct_float function."""

    def test_converts_valid_float(self) -> None:
        """Valid floats are converted correctly."""
        result = correct_float(3.14)
        assert result == 3.14

    def test_converts_string_float(self) -> None:
        """String representations of floats are converted."""
        result = correct_float("2.5")
        assert result == 2.5

    def test_converts_int(self) -> None:
        """Integers are converted to floats."""
        result = correct_float(42)
        assert result == 42.0

    def test_invalid_returns_nan(self) -> None:
        """Invalid values return NaN."""
        result = correct_float("not_a_number")
        assert np.isnan(result)

    def test_vectorized_array(self) -> None:
        """Works on numpy arrays."""
        arr = np.array([1, "2.5", "bad", None])
        result = correct_float(arr)
        assert len(result) == 4
        assert result[0] == 1.0
        assert result[1] == 2.5
        assert np.isnan(result[2])
        assert np.isnan(result[3])


class TestCorrectStr:
    """Tests for correct_str function."""

    def test_converts_string(self) -> None:
        """Strings pass through."""
        assert correct_str("hello") == "hello"

    def test_converts_int_to_str(self) -> None:
        """Integers become strings."""
        assert correct_str(42) == "42"

    def test_none_returns_nan_string(self) -> None:
        """None becomes 'nan'."""
        assert correct_str(None) == "nan"


class TestLinearUnitConverter:
    """Tests for LinearUnitConverter class."""

    def test_feet_to_meters(self) -> None:
        """Converts feet to meters correctly."""
        conv = LinearUnitConverter(0.3048)
        result = conv(1000)
        np.testing.assert_allclose(result, 304.8)

    def test_knots_to_mps(self) -> None:
        """Converts knots to m/s correctly."""
        conv = LinearUnitConverter(1852 / 3600)
        result = conv(100)
        np.testing.assert_allclose(result, 51.44444, rtol=1e-4)

    def test_array_input(self) -> None:
        """Works with numpy arrays."""
        conv = LinearUnitConverter(2.0)
        arr = np.array([1, 2, 3])
        result = conv(arr)
        np.testing.assert_array_equal(result, [2, 4, 6])


class TestAdditionUnitConverter:
    """Tests for AdditionUnitConverter class."""

    def test_celsius_to_kelvin(self) -> None:
        """Converts Celsius to Kelvin."""
        conv = AdditionUnitConverter(273.15)
        result = conv(0)
        np.testing.assert_allclose(result, 273.15)

    def test_array_input(self) -> None:
        """Works with numpy arrays."""
        conv = AdditionUnitConverter(10)
        arr = np.array([0, 5, 10])
        result = conv(arr)
        np.testing.assert_array_equal(result, [10, 15, 20])


class TestCategoryMapper:
    """Tests for CategoryMapper class."""

    def test_maps_categories(self) -> None:
        """Maps string categories to integers."""
        mapper = CategoryMapper({"low": 0, "medium": 1, "high": 2})
        result = mapper(np.array(["low", "high", "medium"]))
        np.testing.assert_array_equal(result, [0, 2, 1])

    def test_unknown_returns_nan(self) -> None:
        """Unknown categories return NaN."""
        mapper = CategoryMapper({"a": 0, "b": 1})
        result = mapper(np.array(["a", "unknown", "b"]))
        assert result[0] == 0
        assert np.isnan(result[1])
        assert result[2] == 1

    def test_inverse_mapping(self) -> None:
        """Inverse mapping works correctly."""
        mapper = CategoryMapper({"left": 0, "right": 1})
        result = mapper.inverse(np.array([0, 1, 0]))
        np.testing.assert_array_equal(result, ["left", "right", "left"])


class TestIdentity:
    """Tests for identity function."""

    def test_returns_same_value(self) -> None:
        """Identity returns input unchanged."""
        assert identity(42) == 42
        assert identity("hello") == "hello"
        assert identity(None) is None
