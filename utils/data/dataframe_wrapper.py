import pandas as pd
from typing import Union, List
from utils.data.column import Column

class IlocIndexer:
    def __init__(self, parent):
        self.parent = parent  # instance de DataFrameWrapper

    def __getitem__(self, key):
        result = self.parent._df.iloc[key]
        if isinstance(result, pd.DataFrame):
            return DataFrameWrapper(result)
        else:
            return result  # Série ou autre, on retourne brut        
        
class LocIndexer:
    def __init__(self, parent):
        self.parent = parent  # instance de DataFrameWrapper

    def __getitem__(self, key):
        result = self.parent._df.loc[key]
        if isinstance(result, pd.DataFrame):
            return DataFrameWrapper(result)
        else:
            return result

    def __setitem__(self, key, value):
        self.parent._df.loc[key] = value

class DataFrameWrapper:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __getitem__(self, key: Union[str, Column, List[Union[str, Column]], pd.Series]):
        # Gestion du filtre booléen pandas
        if isinstance(key, pd.Series) and key.dtype == bool:
            filtered_df = self._df[key]
            return DataFrameWrapper(filtered_df)

        # Gestion liste de colonnes (str ou Column)
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

        # Gestion clé unique Column
        if isinstance(key, Column):
            col_name = key.col_name
            if col_name not in self._df.columns:
                raise KeyError(f"Column '{col_name}' not found in DataFrame")
            return self._df[col_name]

        # Gestion clé unique str
        if isinstance(key, str):
            if key not in self._df.columns:
                raise KeyError(f"Column '{key}' not found in DataFrame")
            return self._df[key]

        raise TypeError(f"Unsupported key type: {type(key)}")

    def __setitem__(self, key: Union[str, Column, List[Union[str, Column]]], value):
        # Gestion liste de colonnes
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

        # Gestion clé unique Column
        elif isinstance(key, Column):
            self._df[key.col_name] = value

        # Gestion clé unique str
        elif isinstance(key, str):
            self._df[key] = value

        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def __getattr__(self, attr):
        # Délègue les autres attributs/méthodes au DataFrame pandas interne
        return getattr(self._df, attr)

    def __len__(self):
        return len(self._df)

    def bfill(self, *args, **kwargs):
        df_bfilled = self._df.bfill(*args, **kwargs)
        return DataFrameWrapper(df_bfilled)

    def ffill(self, *args, **kwargs):
        df_ffilled = self._df.ffill(*args, **kwargs)
        return DataFrameWrapper(df_ffilled)

    @property
    def iloc(self):
        return IlocIndexer(self)

    @property
    def loc(self):
        return LocIndexer(self)
