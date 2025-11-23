from utils.data.column import Column
from utils.data.dataframe_wrapper import DataFrameWrapper

class FlightProcessor:
    """
    Flexible flight data processor with a customizable post-processing hook.
    """

    def __init__(self, model_cols, custom_processing_fn=None):
        """
        Parameters
        ----------
        model_cols : tuple
            (x_cols, u_cols, e0_cols, e_cols, dx_cols)
        custom_processing_fn : callable, optional
            Function that takes (df) and returns (df),
            allowing the user to add custom logic after base processing.
        """
        (
            self.x_cols,
            self.u_cols,
            self.e0_cols,
            self.e_cols,
            self.dx_cols,
        ) = model_cols
        self.dx_cols = [el[1] for el in self.dx_cols]
        self.custom_processing_fn = custom_processing_fn

    # ------------------------------------------------------------------
    def process_flight(self, df):
        """Main flight preprocessing pipeline."""

        df = DataFrameWrapper(df)

        # === 1. Base column standardization ===
        for col in Column.get_all():
            raw_col = col.raw_name
            gold_col = col.col_name
            if raw_col is not None and raw_col in df.columns:
                df[gold_col] = col.unit.convert(df[raw_col])

        # === 2. Automatic derivative computation ===
        for col, d_col in zip(self.x_cols, self.dx_cols):
            df[d_col] = df[col].diff(1).bfill()

        # === 3. Optional user-defined processing ===
        if self.custom_processing_fn is not None:
            df = self.custom_processing_fn(df)

        # === 4. Return only model-relevant columns ===
        return df[self.x_cols + self.u_cols + self.e0_cols + self.e_cols + self.dx_cols]
