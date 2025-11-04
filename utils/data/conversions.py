import numpy as np
import pandas as pd


@np.vectorize
def correct_float(el):
    """
    Convert element to float, return np.nan if conversion fails.

    Args:
        el: Input element.

    Returns:
        float or np.nan
    """
    try:
        return float(el)
    except Exception:
        return np.nan


@np.vectorize
def correct_str(el):
    """
    Convert element to string, return 'nan' string if None.

    Args:
        el: Input element.

    Returns:
        str
    """
    if el is None:
        return "nan"
    return str(el)


def map_cat_dict(cat_dict):
    """
    Create a vectorized mapping function from a category dictionary.

    Args:
        cat_dict (dict): Mapping dictionary.

    Returns:
        function: Vectorized function mapping input elements via cat_dict.
    """

    @np.vectorize
    def map_dict(el):
        return cat_dict[el]

    return map_dict


class CategoryMapper:
    def __init__(self, cat_dict):
        self.cat_dict = cat_dict
        self.inv_cat_dict = {v: k for k, v in cat_dict.items()}  # Build inverse dict
        self.inv_vectorized = np.vectorize(self._inv_map)

    def _map(self, el):
        try:
            return self.cat_dict[el]
        except KeyError:
            return np.nan
    
    def _inv_map(self, el):
        return self.inv_cat_dict[el]

    def __call__(self, array_like):
        vectorized = np.vectorize(self._map, otypes=[float])
        return vectorized(array_like)
    
    def inverse(self, array_like):
        return self.inv_vectorized(array_like)




def identity(value):
    """
    Identity function that returns input value unchanged.

    Args:
        value: Any input.

    Returns:
        Same as input.
    """
    return value

    

def linear_unit_conversion(unit):
    """
    Returns a function that converts a value by multiplying by a unit factor.

    Args:
        unit (float): Conversion factor.

    Returns:
        function: Function to apply linear conversion.
    """
    return lambda value: value * unit


class LinearUnitConverter:
    def __init__(self, unit):
        self.unit = unit

    def __call__(self, value):
        return value * self.unit



def addition_unit_conversion(unit):
    """
    Returns a function that converts a value by adding a unit offset.

    Args:
        unit (float): Offset to add.

    Returns:
        function: Function to apply addition conversion.
    """
    return lambda value: value + unit



class AdditionUnitConverter:
    def __init__(self, unit):
        self.unit = unit

    def __call__(self, value):
        return value + self.unit



def correct_float_col(df_col):
    """
    Apply correct_float vectorized conversion to a pandas Series or DataFrame column.

    Args:
        df_col (pd.Series): Input column.

    Returns:
        pd.Series: Converted column.
    """
    return df_col.apply(correct_float)


def deg_to_rad_correct_float(df_col):
    """
    Convert degrees to radians on a pandas Series after float correction.

    Args:
        df_col (pd.Series): Input degrees column.

    Returns:
        pd.Series: Column in radians.
    """
    return np.deg2rad(df_col.apply(correct_float))


def unwrap_deg_to_rad_correct_float(df_col):
    """
    Convert degrees to radians and unwrap angles to prevent discontinuities.

    Args:
        df_col (pd.Series): Input degrees column.

    Returns:
        np.ndarray: Unwrapped radians array.
    """
    return np.unwrap(deg_to_rad_correct_float(df_col))


def mapping(dict_map):
    """
    Returns a function that maps pandas Series values using a dictionary.

    Args:
        dict_map (dict): Mapping dictionary.

    Returns:
        function: Function to apply mapping to Series.
    """
    return lambda df_col: df_col.map(dict_map)





def one_hot_encoding(df_col, dim):
    """
    One-hot encode a categorical pandas Series with fixed categories.

    Args:
        df_col (pd.Series): Categorical input column.
        dim (int): Number of categories (dimensions).

    Returns:
        pd.DataFrame: One-hot encoded DataFrame.
    """
    categories = list(range(dim))
    df_col_cat = pd.Categorical(df_col, categories=categories)
    one_hot = pd.get_dummies(df_col_cat, prefix=df_col.name + "_one_hot").astype(int)
    return one_hot
