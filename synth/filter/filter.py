from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T = TypeVar("T")


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

    def __neg__(self) -> "Filter[T]":
        return self.complementary()

    def complementary(self) -> "Filter[T]":
        return NegFilter(self)


class NegFilter(Filter, Generic[T]):
    def __init__(self, filter: Filter[T]) -> None:
        self.filter = filter

    def accept(self, obj: T) -> bool:
        return not self.filter.accept(obj)

    def complementary(self) -> "Filter[T]":
        return self.filter


class UnionFilter(Filter, Generic[T]):
    def __init__(self, *filters: Filter[T]) -> None:
        self.filters = list(filters)

    def accept(self, obj: T) -> bool:
        return any(p.accept(obj) for p in self.filters)


class IntersectionFilter(Filter, Generic[T]):
    def __init__(self, *filters: Filter[T]) -> None:
        self.filters = list(filters)

    def accept(self, obj: T) -> bool:
        return all(p.accept(obj) for p in self.filters)
