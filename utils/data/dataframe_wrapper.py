#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Light wrapper around pandas DataFrame adding Column-aware accessors."""

from typing import Any, List, Union

import pandas as pd
from utils.data.column import Column

class IlocIndexer:
    """Custom iloc indexer returning wrapped DataFrames."""

    def __init__(self, parent):
        """Store reference to parent wrapper."""
        self.parent = parent  # instance de DataFrameWrapper

    def __getitem__(self, key):
        """Retrieve by integer-location indexing; wrap DataFrame results."""
        result = self.parent._df.iloc[key]
        if isinstance(result, pd.DataFrame):
            return DataFrameWrapper(result)
        else:
            return result  # Série ou autre, on retourne brut        
        
class LocIndexer:
    """Custom loc indexer returning wrapped DataFrames."""

    def __init__(self, parent):
        """Store reference to parent wrapper."""
        self.parent = parent  # instance de DataFrameWrapper

    def __getitem__(self, key):
        """Retrieve by label-based indexing; wrap DataFrame results."""
        result = self.parent._df.loc[key]
        if isinstance(result, pd.DataFrame):
            return DataFrameWrapper(result)
        else:
            return result

    def __setitem__(self, key, value):
        """Assign by label-based indexing."""
        self.parent._df.loc[key] = value

class DataFrameWrapper:
    """Wrapper to allow Column objects in DataFrame indexing and assignment."""

    def __init__(self, df: pd.DataFrame):
        """Store the underlying pandas DataFrame."""
        self._df = df

    def __getitem__(self, key: Union[str, Column, List[Union[str, Column]], pd.Series]):
        """Enhanced getter supporting Column objects and boolean masks.

        Args:
            key: Column name, Column instance, list of columns, or boolean mask.

        Returns:
            DataFrameWrapper for multi-column selection, Series for single-column selection, or filtered wrapper for masks.
        """
        if isinstance(key, pd.Series) and key.dtype == bool:
            filtered_df = self._df[key]
            return DataFrameWrapper(filtered_df)

        if isinstance(key, list):
            col_names = []
            for k in key:
                if isinstance(k, Column):
                    col_name = k.col_name
                elif isinstance(k, str):
                    col_name = k
                else:
                    raise TypeError(f"Unsupported key type in list: {type(k)}")
                if col_name not in self._df.columns:
                    raise KeyError(f"Column '{col_name}' not found in DataFrame")
                col_names.append(col_name)
            return DataFrameWrapper(self._df[col_names])

        if isinstance(key, Column):
            col_name = key.col_name
            if col_name not in self._df.columns:
                raise KeyError(f"Column '{col_name}' not found in DataFrame")
            return self._df[col_name]

        if isinstance(key, str):
            if key not in self._df.columns:
                raise KeyError(f"Column '{key}' not found in DataFrame")
            return self._df[key]

        raise TypeError(f"Unsupported key type: {type(key)}")

    def __setitem__(self, key: Union[str, Column, List[Union[str, Column]]], value) -> None:
        """Assign values using Column-aware keys."""
        if isinstance(key, list):
            col_names = []
            for k in key:
                if isinstance(k, Column):
                    col_name = k.col_name
                elif isinstance(k, str):
                    col_name = k
                else:
                    raise TypeError(f"Unsupported key type in list: {type(k)}")
                col_names.append(col_name)

            if hasattr(value, '__getitem__'):
                for i, col in enumerate(col_names):
                    if isinstance(value, (list, tuple)):
                        self._df[col] = value[i]
                    elif col in value:
                        self._df[col] = value[col]
                    else:
                        self._df[col] = value
            else:
                raise TypeError("Value must be indexable (list, tuple, dict, DataFrame) when key is a list")

        elif isinstance(key, Column):
            self._df[key.col_name] = value

        elif isinstance(key, str):
            self._df[key] = value

        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def __getattr__(self, attr) -> Any:
        """Delegate missing attributes to the underlying pandas DataFrame."""
        return getattr(self._df, attr)

    def __len__(self) -> int:
        """Return number of rows."""
        return len(self._df)

    def bfill(self, *args, **kwargs) -> "DataFrameWrapper":
        """Return a backfilled DataFrameWrapper."""
        df_bfilled = self._df.bfill(*args, **kwargs)
        return DataFrameWrapper(df_bfilled)

    def ffill(self, *args, **kwargs) -> "DataFrameWrapper":
        """Return a forward-filled DataFrameWrapper."""
        df_ffilled = self._df.ffill(*args, **kwargs)
        return DataFrameWrapper(df_ffilled)

    @property
    def iloc(self):
        """Expose integer-location based indexer."""
        return IlocIndexer(self)

    @property
    def loc(self):
        """Expose label-based indexer."""
        return LocIndexer(self)
