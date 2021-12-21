import os
from typing import Any, Callable, List


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
