from typing import Any, Generator, List, TypeVar

T = TypeVar("T")


def gen_take(gen: Generator[T, Any, Any], n: int) -> List[T]:
    """
    Take the first n elements of a generator and return them as a list.
    """
    out: List[T] = []
    try:
        for _ in range(n):
            out.append(next(gen))
    except StopIteration:
        pass
    return out
