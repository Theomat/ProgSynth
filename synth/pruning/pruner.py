from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class Pruner(ABC, Generic[T]):
    @abstractmethod
    def accept(self, obj: T) -> bool:
        pass


class UnionPruner(Pruner[U]):
    def __init__(self, pruners: Iterable[Pruner[U]]) -> None:
        self.pruners = list(pruners)

    def accept(self, obj: U) -> bool:
        return all(p.accept(obj) for p in self.pruners)
