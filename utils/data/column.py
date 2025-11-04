from dataclasses import dataclass
from utils.data.unit import Unit
from typing import Callable, Optional, ClassVar, List, Dict

@dataclass
class Column:
    """
    Represents a feature column with associated metadata.

    Attributes:
        gold_name (str): Canonical or "gold" name of the column.
        alias (str): Short alias used for referencing the column.
        raw_name (str): Original/raw name of the column from source data.
        unit (Unit): Unit information associated with the column.
        normalize_mode (str): Mode for normalization. Defaults to "normal".
        denormalize_mode (str): Mode for denormalization. Defaults to "normal_clamp".
        last_activation_fn (Optional[Callable]): Optional activation function applied last.
    """

    gold_name: str
    alias: str
    raw_name: str
    unit: Unit
    normalize_mode: str = "normal"
    denormalize_mode: str = "normal_clamp"
    last_activation_fn: Optional[Callable] = None
    loss_name : str = "mse"

    # Class variable to store instances by alias
    _instances_dict: ClassVar[Dict[str, "Column"]] = {}

    def __post_init__(self) -> None:
        """
        Post-initialization to register the instance and generate display properties.
        """
        # Register instance keyed by alias to avoid duplicates
        self._instances_dict[self.alias] = self

        # Prepare display name parts
        abbr = self.unit.si_abbr
        display_list = [word.capitalize() for word in self.gold_name.split(" ")]
        col_list = [self.alias]

        if abbr is not None:
            # Append unit abbreviation nicely formatted
            display_list.append(f"({abbr})")
            # Create a simplified column name by cleaning abbreviation
            col_list.append(abbr.replace("/", "").replace("%", "pct").replace("²", "2").lower())

        # Human-friendly display name, e.g., "Altitude (m)"
        self.display_name: str = " ".join(display_list)
        # Programmatic column name, e.g., "altitude_m"
        self.col_name: str = "_".join(col_list)
            
    def __str__(self) -> str:
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
        """
        Generate a Column instance representing the derivative of this column.

        Returns:
            Column: A new Column instance representing the derivative.
        """
        return Column(
            gold_name=f"derivative_{self.gold_name}",
            alias=f"deriv_{self.alias}",
            raw_name=None,
            unit=self.unit.deriv_unit,
        )

    @classmethod
    def get_all(cls) -> List["Column"]:
        """
        Retrieve all registered Column instances.

        Returns:
            List[Column]: List of all Column instances stored.
        """
        return list(cls._instances_dict.values())
    
    
    def __hash__(self):
        return hash(self.col_name)

    def __eq__(self, other):
        if not isinstance(other, Column):
            return False
        return self.col_name == other.col_name
    