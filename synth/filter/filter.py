from abc import ABC, abstractmethod
from types import UnionType
from typing import Any, Generic, TypeVar


T = TypeVar("T")
U = TypeVar("U")


class Filter(ABC, Generic[T]):
    @abstractmethod
    def accept(self, obj: T) -> bool:

        """
        Accepts objects that should be kept.
        """
        pass

    def reject(self, obj: T) -> bool:
        """
        Rejects objects that should NOT be kept.
        """
        return not self.accept(obj)

    def __and__(self, other: "Filter[T]") -> "IntersectionFilter[T]":
        return self.intersection(other)

    def intersection(self, other: "Filter[T]") -> "IntersectionFilter[T]":
        if isinstance(other, IntersectionFilter):
            return other.intersection(self)
        elif isinstance(self, IntersectionFilter):
            if isinstance(other, IntersectionFilter):
                return IntersectionFilter(*self.filters, *other.filters)
            return IntersectionFilter(*self.filters, other)
        else:
            return IntersectionFilter(self, other)

    def __or__(self, other: "Filter[T]") -> "UnionFilter[T]":
        return self.union(other)

    def union(self, other: "Filter[T]") -> "UnionFilter[T]":
        if isinstance(other, UnionFilter):
            return other.union(self)
        elif isinstance(self, UnionFilter):
            if isinstance(other, UnionFilter):
                return UnionFilter(*self.filters, *other.filters)
            return UnionFilter(*self.filters, other)
        else:
            return UnionFilter(self, other)


class UnionFilter(Filter, Generic[U]):
    def __init__(self, *filters: Filter[U]) -> None:
        self.filters = list(filters)

    def accept(self, obj: U) -> bool:
        return any(p.accept(obj) for p in self.filters)


class IntersectionFilter(Filter, Generic[U]):
    def __init__(self, *filters: Filter[U]) -> None:
        self.filters = list(filters)

    def accept(self, obj: U) -> bool:
        return all(p.accept(obj) for p in self.filters)
