#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conversion helpers for data cleaning and unit manipulation."""

from typing import Any, Dict, Callable

import numpy as np
import pandas as pd


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


def map_cat_dict(cat_dict: Dict[Any, Any]) -> Callable[[Any], Any]:
    """Create a vectorized mapping function from a category dictionary.

    Args:
        cat_dict: Mapping from original to target category values.

    Returns:
        Vectorized function applying the mapping.
    """

    @np.vectorize
    def map_dict(el):
        return cat_dict[el]

    return map_dict


class CategoryMapper:
    """Helper to map categories and their inverse with vectorized functions."""

    def __init__(self, cat_dict: Dict[Any, Any]):
        """Initialize mapper with forward and inverse dictionaries.

        Args:
            cat_dict: Mapping from labels to numeric codes (or any target values).
        """
        self.cat_dict = cat_dict
        self.inv_cat_dict = {v: k for k, v in cat_dict.items()}  # Build inverse dict
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
        return vectorized(array_like)
    
    def inverse(self, array_like: Any) -> np.ndarray:
        """Map category codes back to original labels.

        Args:
            array_like: Array-like of codes to convert back to labels.

        Returns:
            NumPy array of original labels.
        """
        return self.inv_vectorized(array_like)




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
        self.unit = unit

    def __call__(self, value: Any) -> Any:
        """Add the configured offset to the provided value."""
        return value + self.unit



def correct_float_col(df_col: pd.Series) -> pd.Series:
    """Apply correct_float vectorized conversion to a pandas Series or DataFrame column.

    Args:
        df_col: Input column.

    Returns:
        Column converted to floats with failures as ``np.nan``.
    """
    return df_col.apply(correct_float)


def deg_to_rad_correct_float(df_col: pd.Series) -> pd.Series:
    """Convert degrees to radians on a pandas Series after float correction.

    Args:
        df_col: Degrees column.

    Returns:
        Column in radians.
    """
    return np.deg2rad(df_col.apply(correct_float))


def unwrap_deg_to_rad_correct_float(df_col: pd.Series) -> np.ndarray:
    """Convert degrees to radians and unwrap angles to prevent discontinuities.

    Args:
        df_col: Degrees column.

    Returns:
        Unwrapped radians array.
    """
    return np.unwrap(deg_to_rad_correct_float(df_col))


def mapping(dict_map: Dict[Any, Any]) -> Callable[[pd.Series], pd.Series]:
    """Return a function that maps pandas Series values using a dictionary.

    Args:
        dict_map: Mapping from old to new values.

    Returns:
        Callable applying the mapping to a Series.
    """
    return lambda df_col: df_col.map(dict_map)


def one_hot_encoding(df_col: pd.Series, dim: int) -> pd.DataFrame:
    """One-hot encode a categorical pandas Series with fixed categories.

    Args:
        df_col: Input categorical column.
        dim: Number of categories.

    Returns:
        One-hot encoded DataFrame.
    """
    categories = list(range(dim))
    df_col_cat = pd.Categorical(df_col, categories=categories)
    one_hot = pd.get_dummies(df_col_cat, prefix=df_col.name + "_one_hot").astype(int)
    return one_hot
