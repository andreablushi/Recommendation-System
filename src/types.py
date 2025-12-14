from dataclasses import dataclass
from typing import Union, Literal

@dataclass(frozen=True)
class BooleanCondition:
    """
    A condition representing a boolean feature check.
    feature: str - The name of the feature.
    value: bool - The boolean value must be assigned to the feature to satisfy the condition.
    """
    feature: str
    value: bool

@dataclass(frozen=True)
class ThresholdCondition:
    """
    A condition representing a threshold feature check.
    feature: str - The name of the feature.
    op: Literal["<=", ">"] - The comparison operator.
    threshold: float - The threshold value to compare against.
    """
    feature: str
    op: Literal["<=", ">"]
    threshold: float

# A condition can be either a BooleanCondition or a ThresholdCondition
Condition = Union[BooleanCondition, ThresholdCondition]
# A path is a list of conditions
Path = list[Condition]
