import os
from typing import Any, Callable, Generator, List, TypeVar


def make_deterministic_hash() -> None:
    """
    Make sure Python hash are deterministic.
    """
    hashseed = os.getenv("PYTHONHASHSEED")
    if not hashseed:
        os.environ["PYTHONHASHSEED"] = "0"


def to_partial_fun(f: Callable, nargs: int) -> Callable:
    """
    Transform a n-ary function into a function that take n arguments sequentially.
    ```python
    >>> def plus(a, b):
    ...     return a + b
    >>> to_partial_fun(plus, 2)
    lambda a: lambda b: a + b
    >>> # Not exactly equal to but is semantically equivalent to
    ```
    """

    def partial(x: Any, carry_on: List) -> Any:
        li = carry_on[:]
        li.append(x)
        if len(li) == nargs:
            return f(*li)
        return lambda y: partial(y, li)

    return lambda x: partial(x, [])


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
