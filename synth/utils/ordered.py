from typing import Any, Protocol, List
from abc import abstractmethod


class Ordered(Protocol):
    @abstractmethod
    def __lt__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __gt__(self, other: Any) -> bool:
        pass


class Bucket(Ordered):
    def __init__(self, tup: List[int] = [0, 0, 0]):
        self.elems = []
        self.size = len(tup)
        for elem in tup:
            self.elems.append(elem)

    def __str__(self) -> str:
        s = "("
        for elem in self.elems:
            s += "{},".format(elem)
        s = s[:-1] + ")"
        return s

    def __repr__(self) -> str:
        return str(self)

    def __lt__(self, other: "Bucket") -> bool:
        if self.size == 0:
            return False

        if self.elems[0] > other.elems[0]:
            return True
        elif self.elems[0] == other.elems[0]:
            return Bucket(self.elems[1:]).__lt__(Bucket(other.elems[1:]))
        else:
            return False

    def __gt__(self, other: "Bucket") -> bool:
        return other.__lt__(self)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Bucket) and all(
            self.elems[i] == other.elems[i] for i in range(self.size)
        )

    def __le__(self, other: Any) -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other: Any) -> bool:
        return self.__gt__(other) or self.__eq__(other)

    def __neg__(self) -> "Bucket":
        new_elems = [-elem for elem in self.elems]
        return Bucket(new_elems)

    def __iadd__(self, other: "Bucket") -> "Bucket":
        if self.size == other.size:
            for i in range(self.size):
                self.elems[i] += other.elems[i]
            return self
        else:
            raise RuntimeError(
                "size mismatch, Bucket{}: {}, Bucket{}: {}".format(
                    self, self.size, other, other.size
                )
            )

    def add_prob_uniform(self, probability: float) -> None:
        """
        Given a probability add 1 in the relevant bucket assuming buckets are linearly distributed.
        """
        index = self.size - int(probability * self.size) - 1
        self.elems[index] += 1
