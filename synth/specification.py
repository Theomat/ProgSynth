from dataclasses import dataclass
from typing import Any, List

from synth.syntax.type_system import FunctionType, Type, guess_type


class TaskSpecification:
    pass


@dataclass
class Example:
    """
    Represents an example pair of (inputs, output) 
    """
    inputs: List[Any]
    output: Any

    def guess_type(self) -> Type:
        types = list(map(guess_type, self.inputs)) + [guess_type(self.output)]
        return FunctionType(*types)


@dataclass
class PBE(TaskSpecification):
    """
    Programming By Example (PBE) specification.
    """
    examples: List[Example]

    def guess_type(self) -> Type:
        return self.examples[0].guess_type()
