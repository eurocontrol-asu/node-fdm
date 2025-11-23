#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flight preprocessing pipeline for converting raw data into model-ready columns."""

from typing import Any, Callable, Optional, Tuple

from utils.data.column import Column
from utils.data.dataframe_wrapper import DataFrameWrapper


class FlightProcessor:
    """Flexible flight data processor with a customizable post-processing hook."""

    def __init__(
        self,
        model_cols: Tuple[Any, Any, Any, Any, Any],
        custom_processing_fn: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Initialize the processor with model column configuration and hooks.

        Args:
            model_cols: Tuple of model column groups (state, control, env, etc.).
            custom_processing_fn: Optional callable applied after base processing; uses Any for flexibility with DataFrame-like inputs.
        """
        (
            self.x_cols,
            self.u_cols,
            self.e0_cols,
            self.e_cols,
            self.dx_cols,
        ) = model_cols
        self.dx_cols = [col.derivative for col in self.x_cols]
        self.custom_processing_fn = custom_processing_fn

    # ------------------------------------------------------------------
    def process_flight(self, df: Any) -> DataFrameWrapper:
        """Run the main flight preprocessing pipeline.

        Args:
            df: DataFrame-like object containing raw flight data. Uses Any for flexibility across wrappers.

        Returns:
            Processed DataFrameWrapper filtered to model-relevant columns.
        """

        df = DataFrameWrapper(df)

        for col in Column.get_all():
            raw_col = col.raw_name
            gold_col = col.col_name
            if raw_col is not None and raw_col in df.columns:
                df[gold_col] = col.unit.convert(df[raw_col])

        for col in self.x_cols:
            df[col.derivative] = df[col].diff(1).bfill()

        if self.custom_processing_fn is not None:
            df = self.custom_processing_fn(df)

        return df[self.x_cols + self.u_cols + self.e0_cols + self.e_cols + self.dx_cols]
