from typing import Callable, Optional
from dataclasses import dataclass

from utils.data.conversions import (
    identity, 
    correct_float,
    correct_str,
)

@dataclass
class Unit:
    """
    Represents a physical unit with conversion to SI units.

    Attributes:
        name (str): Full name of the unit (e.g., "meter per second").
        abbr (str): Unit abbreviation (e.g., "m/s").
        value_type (str): Type of value, used to select correction function ('float' or 'str').
        si_unit (Optional[Unit]): Corresponding SI unit (can be self).
        derivative (Optional[Unit]): Unit representing the derivative (e.g., m/s² for m/s).
        modifier (Callable): Function to convert value to SI unit.
    """

    name: str
    abbr: str
    value_type: str = 'float'
    si_unit: Optional["Unit"] = None
    derivative: Optional["Unit"] = None
    modifier: Callable = identity

    def convert(self, value):
        """
        Convert a value in this unit to the corresponding SI unit value.

        Args:
            value: The value to convert.

        Returns:
            Converted value in SI units.
        """
        # Choose the appropriate correction function based on value_type
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
        """
        Get the abbreviation of the SI unit.

        Returns:
            str: Abbreviation of the SI unit if defined, else own abbreviation.
        """
        if self.si_unit is not None:
            return self.si_unit.abbr
        return self.abbr

    @property
    def deriv_unit(self) -> Optional["Unit"]:
        """
        Get the derivative unit.

        Returns:
            Unit or None: The derivative unit if defined.
        """
        if self.si_unit is not None and self.si_unit.derivative is not None:
            return self.si_unit.derivative
        return self.derivative
