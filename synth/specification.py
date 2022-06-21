from dataclasses import dataclass
from typing import Any, List

from synth.syntax.type_system import FunctionType, EmptyList, Type, guess_type


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
        i = 0
        t = self.examples[i].guess_type()
        while EmptyList in t and i + 1 < len(self.examples):
            i += 1
            t = self.examples[i].guess_type()
        return t

@dataclass
class NLP(TaskSpecification):
    """
    Natural Language (NLP) specification.
    """

    intent: str

@dataclass
class NLPBE(TaskSpecification):
    """
    Combined Natural Language and Example specifications.
    """

    intent: str
    examples: List[Example]

    def guess_type(self) -> Type:
        i = 0
        t = self.examples[i].guess_type()
        while EmptyList in t and i + 1 < len(self.examples):
            i += 1
            t = self.examples[i].guess_type()
        return t
