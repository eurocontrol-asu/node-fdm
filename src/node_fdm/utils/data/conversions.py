#!/usr/bin/env python3
"""Conversion helpers for data cleaning and unit manipulation.

This module provides numpy-based utilities for:
- Type correction (float, string)
- Unit conversions (linear, additive)
- Category mapping
"""

from collections.abc import Callable
from typing import Any

import numpy as np


@np.vectorize
def correct_float(el: Any) -> float:
    """Convert element to float; return ``np.nan`` on failure.

    Args:
        el: Input element to convert.

    Returns:
        Floating-point value or ``np.nan`` when conversion fails.
    """
    try:
        return float(el)
    except Exception:
        return np.nan


@np.vectorize
def correct_str(el: Any) -> str:
    """Convert element to string; return ``'nan'`` if None.

    Args:
        el: Input element to convert.

    Returns:
        String representation or ``'nan'`` when input is None.
    """
    if el is None:
        return "nan"
    return str(el)


def map_cat_dict(cat_dict: dict[Any, Any]) -> Callable[[Any], Any]:
    """Create a vectorized mapping function from a category dictionary.

    Args:
        cat_dict: Mapping from original to target category values.

    Returns:
        Vectorized function applying the mapping.
    """

    @np.vectorize
    def map_dict(el: Any) -> Any:
        return cat_dict[el]

    return map_dict


class CategoryMapper:
    """Helper to map categories and their inverse with vectorized functions."""

    def __init__(self, cat_dict: dict[Any, Any]) -> None:
        """Initialize mapper with forward and inverse dictionaries.

        Args:
            cat_dict: Mapping from labels to numeric codes (or any target values).
        """
        self.cat_dict = cat_dict
        self.inv_cat_dict = {v: k for k, v in cat_dict.items()}
        self.inv_vectorized = np.vectorize(self._inv_map)

    def _map(self, el: Any) -> Any:
        """Map a single element to its numeric code."""
        try:
            return self.cat_dict[el]
        except KeyError:
            return np.nan

    def _inv_map(self, el: Any) -> Any:
        """Map a numeric code back to its original label."""
        return self.inv_cat_dict[el]

    def __call__(self, array_like: Any) -> np.ndarray:
        """Map values to category codes.

        Args:
            array_like: Array-like of labels to convert.

        Returns:
            NumPy array of mapped codes (``np.nan`` for unknown labels).
        """
        vectorized = np.vectorize(self._map, otypes=[float])
        result: np.ndarray[Any, np.dtype[Any]] = vectorized(array_like)
        return result

    def inverse(self, array_like: Any) -> np.ndarray:
        """Map category codes back to original labels.

        Args:
            array_like: Array-like of codes to convert back to labels.

        Returns:
            NumPy array of original labels.
        """
        result: np.ndarray[Any, np.dtype[Any]] = self.inv_vectorized(array_like)
        return result


def identity(value: Any) -> Any:
    """Return input value unchanged.

    Args:
        value: Any input.

    Returns:
        Same value passed in.
    """
    return value


def linear_unit_conversion(unit: float) -> Callable[[Any], Any]:
    """Return a function that multiplies values by a unit factor.

    Args:
        unit: Scaling factor to apply.

    Returns:
        Callable that multiplies inputs by ``unit``.
    """
    return lambda value: value * unit


class LinearUnitConverter:
    """Apply a linear scaling factor to values."""

    def __init__(self, unit: float) -> None:
        """Initialize with scaling factor.

        Args:
            unit: Factor to multiply values by.
        """
        self.unit = unit

    def __call__(self, value: Any) -> Any:
        """Multiply value by the configured unit factor."""
        return value * self.unit


def addition_unit_conversion(unit: float) -> Callable[[Any], Any]:
    """Return a function that adds a unit offset to values.

    Args:
        unit: Offset to add.

    Returns:
        Callable that adds ``unit`` to inputs.
    """
    return lambda value: value + unit


class AdditionUnitConverter:
    """Apply an additive offset to values."""

    def __init__(self, unit: float) -> None:
        """Initialize with offset value.

        Args:
            unit: Offset to add to values.
        """
        self.unit = unit

    def __call__(self, value: Any) -> Any:
        """Add the configured offset to the provided value."""
        return value + self.unit
