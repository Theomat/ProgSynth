from abc import abstractmethod, ABC
from typing import Generic, TypeVar

from synth.specification import TaskSpecification
from synth.task import Task

T = TypeVar("T", bound=TaskSpecification, covariant=True)
U = TypeVar("U")


class SpecificationEncoder(ABC, Generic[T, U]):
    @abstractmethod
    def encode(self, task: Task[T]) -> U:
        pass
