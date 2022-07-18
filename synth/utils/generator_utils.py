from typing import Any, Generator, List, TypeVar

from tqdm import trange
T = TypeVar("T")


def gen_take(gen: Generator[T, Any, Any], n: int, progress: bool = False) -> List[T]:
    """
    Take the first n elements of a generator and return them as a list.
    """
    out: List[T] = []
    try:
        ite = range(n) if not progress else trange(n)
        for _ in ite:
            out.append(next(gen))
    except StopIteration:
        pass
    return out
