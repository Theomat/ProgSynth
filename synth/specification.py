from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar

from synth.syntax.type_system import EmptyList, Type
from synth.syntax.type_helper import FunctionType, guess_type


class TaskSpecification:
    def get_specification(self, specification: type) -> "Optional[TaskSpecification]":
        """
        Gets the specification of the given TaskSpecification type that may be embed in this possibly compound specification.
        If none is found returns None.
        """
        if isinstance(self, specification):
            return self
        return None


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
class PBEWithConstants(PBE):
    """
    Programming By Example (PBE) with constants specification
    """

    constants: Dict[Type, List[Any]]


@dataclass
class NLP(TaskSpecification):
    """
    Natural Language (NLP) specification.
    """

    intent: str


@dataclass
class SketchedSpecification(TaskSpecification):
    sketch: str


U = TypeVar("U", bound=TaskSpecification, covariant=True)
V = TypeVar("V", bound=TaskSpecification, covariant=True)


@dataclass
class CompoundSpecification(TaskSpecification, Generic[U, V]):
    specification1: U
    specification2: V

    def get_specification(self, specification: type) -> "Optional[TaskSpecification]":
        a = self.specification1.get_specification(specification)
        if a is not None:
            return a
        return self.specification2.get_specification(specification)


NLPBE = CompoundSpecification[NLP, PBE]
SketchedPBE = CompoundSpecification[SketchedSpecification, PBE]
