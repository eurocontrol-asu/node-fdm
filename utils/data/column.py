#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Column metadata definition with unit and normalization helpers."""

from dataclasses import dataclass
from utils.data.unit import Unit
from typing import Callable, Optional, ClassVar, List, Dict

@dataclass
class Column:
    """Represent a feature column with associated metadata.

    Args:
        gold_name: Canonical name of the column.
        alias: Short alias used for references.
        raw_name: Original/raw name from source data.
        unit: Unit metadata associated with the column.
        normalize_mode: Normalization mode to apply.
        denormalize_mode: Denormalization mode to apply.
        last_activation_fn: Optional activation function applied last.
        loss_name: Loss identifier associated with the column.
    """

    gold_name: str
    alias: str
    raw_name: str
    unit: Unit
    normalize_mode: str = "normal"
    denormalize_mode: str = "normal_clamp"
    last_activation_fn: Optional[Callable] = None
    loss_name : str = "mse"

    _instances_dict: ClassVar[Dict[str, "Column"]] = {}

    def __post_init__(self) -> None:
        """Register instance and generate display properties."""
        self._instances_dict[self.alias] = self

        abbr = self.unit.si_abbr
        display_list = [word.capitalize() for word in self.gold_name.split(" ")]
        col_list = [self.alias]

        if abbr is not None:
            display_list.append(f"({abbr})")
            col_list.append(abbr.replace("/", "").replace("%", "pct").replace("²", "2").lower())

        self.display_name: str = " ".join(display_list)
        self.col_name: str = "_".join(col_list)
            
    def __str__(self) -> str:
        """Return the programmatic column name."""
        return self.col_name

    @property
    def name(self) -> str:
        """Return the canonical gold name of the column."""
        return self.gold_name

    @property
    def raw(self) -> str:
        """Return the raw/original name of the column."""
        return self.raw_name

    @property
    def derivative(self) -> "Column":
        """Generate a Column representing the derivative of this column.

        Returns:
            Column: New Column instance for the derivative.
        """
        return Column(
            gold_name=f"derivative_{self.gold_name}",
            alias=f"deriv_{self.alias}",
            raw_name=None,
            unit=self.unit.deriv_unit,
        )

    @classmethod
    def get_all(cls) -> List["Column"]:
        """Retrieve all registered Column instances.

        Returns:
            List[Column]: Registered Column objects.
        """
        return list(cls._instances_dict.values())
    
    
    def __hash__(self):
        """Hash by column name for dict/set usage."""
        return hash(self.col_name)

    def __eq__(self, other):
        """Compare columns by canonical column name."""
        if not isinstance(other, Column):
            return False
        return self.col_name == other.col_name
    
