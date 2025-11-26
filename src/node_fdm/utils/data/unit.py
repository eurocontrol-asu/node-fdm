#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical unit definitions with conversion helpers."""

from typing import Any, Callable, Optional
from dataclasses import dataclass

from utils.data.conversions import (
    identity,
    correct_float,
    correct_str,
)


@dataclass
class Unit:
    """Represent a physical unit with conversion to SI units.

    Args:
        name: Full name of the unit.
        abbr: Unit abbreviation.
        value_type: Type of stored values ("float" or "str") to pick correction.
        si_unit: Corresponding SI unit (may be self).
        derivative: Derivative unit (e.g., m/s² for m/s).
        modifier: Callable to convert values to SI.
    """

    name: str
    abbr: str
    value_type: str = "float"
    si_unit: Optional["Unit"] = None
    derivative: Optional["Unit"] = None
    modifier: Callable = identity

    def convert(self, value: Any) -> Any:
        """Convert a value in this unit to the corresponding SI unit value.

        Args:
            value: Input value to convert (scalar or array-like).

        Returns:
            Converted value in SI units.
        """
        if self.value_type == "float":
            correction = correct_float
        elif self.value_type == "str":
            correction = correct_str
        else:
            correction = identity
        corrected_value = correction(value)
        return self.modifier(corrected_value)

    @property
    def si_abbr(self) -> str:
        """Return the abbreviation of the SI unit if defined, else own abbreviation.

        Returns:
            Abbreviation string.
        """
        if self.si_unit is not None:
            return self.si_unit.abbr
        return self.abbr

    @property
    def deriv_unit(self) -> Optional["Unit"]:
        """Return the derivative unit if defined.

        Returns:
            Derivative `Unit` or None.
        """
        if self.si_unit is not None and self.si_unit.derivative is not None:
            return self.si_unit.derivative
        return self.derivative
