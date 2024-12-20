from functools import wraps
from typing import Callable, Dict, KeysView, List, Optional, Tuple
from types import SimpleNamespace

__COUNTS__: Dict[str, SimpleNamespace] = {}


def __create_count__(name: str):
    __COUNTS__[name] = SimpleNamespace(
        total=0, count=0, max=-float("inf"), min=float("inf")
    )


def __add_entry_to_count__(name: str, value):
    if not name in __COUNTS__:
        __create_count__(name)
    data = __COUNTS__[name]
    data.count += 1
    data.total += value
    data.max = max(data.max, value)
    data.min = min(data.min, value)


def get(name: str) -> SimpleNamespace:
    return __COUNTS__.get(
        name, SimpleNamespace(total=0, count=0, max=-float("inf"), min=float("inf"))
    )


def keys() -> KeysView[str]:
    return __COUNTS__.keys()


def items(domain: str = "") -> List[Tuple[str, SimpleNamespace]]:
    return [(key, entry) for key, entry in __COUNTS__.items() if key.startswith(domain)]


def total(domain: str = "") -> float:
    total_value: float = 0
    if len(domain) > 0:
        domain += "."
    for _, entry in items(domain):
        total_value += entry.total
    return total_value


def count(name: str, value: float):
    return __add_entry_to_count__(name, value)


def countfn(
    _func=None, *, prefix: Optional[str] = None, nanoseconds: Optional[bool] = None
):
    def decorator(func: Callable) -> Callable:
        local_name = func.__name__
        if prefix:
            local_name = prefix + "." + local_name

        @wraps(func)
        def wrapper_func(*args, **kwargs):
            with count(local_name, nanoseconds):
                return func(*args, **kwargs)

        return wrapper_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)
