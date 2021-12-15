from dataclasses import dataclass
from typing import Any, List


class TaskSpecification:
    pass


@dataclass
class Example:
    inputs: List[Any]
    output: Any


@dataclass
class PBE(TaskSpecification):
    examples: List[Example]
