from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T = TypeVar("T")
U = TypeVar("U")


class Pruner(ABC, Generic[T]):
    @abstractmethod
    def accept(self, obj: T) -> bool:
        pass

    def __and__(self, other: "Pruner[T]") -> "IntersectionPruner[T]":
        return self.intersection(other)

    def intersection(self, other: "Pruner[T]") -> "IntersectionPruner[T]":
        if isinstance(other, IntersectionPruner):
            return other.intersection(self)
        elif isinstance(self, IntersectionPruner):
            if isinstance(other, IntersectionPruner):
                return IntersectionPruner(*self.pruners, *other.pruners)
            return IntersectionPruner(*self.pruners, other)
        else:
            return IntersectionPruner(self, other)

    def __or__(self, other: "Pruner[T]") -> "UnionPruner[T]":
        return self.union(other)

    def union(self, other: "Pruner[T]") -> "UnionPruner[T]":
        if isinstance(other, UnionPruner):
            return other.union(self)
        elif isinstance(self, UnionPruner):
            if isinstance(other, UnionPruner):
                return UnionPruner(*self.pruners, *other.pruners)
            return UnionPruner(*self.pruners, other)
        else:
            return UnionPruner(self, other)


class UnionPruner(Pruner, Generic[U]):
    def __init__(self, *pruners: Pruner[U]) -> None:
        self.pruners = list(pruners)

    def accept(self, obj: U) -> bool:
        return any(p.accept(obj) for p in self.pruners)


class IntersectionPruner(Pruner, Generic[U]):
    def __init__(self, *pruners: Pruner[U]) -> None:
        self.pruners = list(pruners)

    def accept(self, obj: U) -> bool:
        return all(p.accept(obj) for p in self.pruners)
