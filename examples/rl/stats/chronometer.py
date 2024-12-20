from functools import wraps
import time

from typing import Callable, Dict, KeysView, List, Optional, Tuple
from types import SimpleNamespace

__CLOCKS__: Dict[str, SimpleNamespace] = {}


def __create_clock__(name: str, nanoseconds: bool):
    __CLOCKS__[name] = SimpleNamespace(
        total=0, count=0, max=0, min=float("inf"), in_nanoseconds=nanoseconds
    )


def __add_entry_to_clock__(name: str, elapsed_time):
    if not name in __CLOCKS__:
        __create_clock__(name)
    data = __CLOCKS__[name]
    data.count += 1
    data.total += elapsed_time
    data.max = max(data.max, elapsed_time)
    data.min = min(data.min, elapsed_time)


def get(name: str) -> SimpleNamespace:
    return __CLOCKS__.get(
        name,
        SimpleNamespace(
            total=0, count=0, max=0, min=float("inf"), in_nanoseconds=False
        ),
    )


def keys() -> KeysView[str]:
    return __CLOCKS__.keys()


def items(domain: str = "") -> List[Tuple[str, SimpleNamespace]]:
    return [(key, entry) for key, entry in __CLOCKS__.items() if key.startswith(domain)]


def total_time(domain: str = "") -> float:
    total_time: float = 0
    if len(domain) > 0:
        domain += "."
    for _, entry in items(domain):
        total_time += entry.total / (1e9 if entry.in_nanoseconds else 1)
    return total_time


class ClockContextManager:
    def __init__(self, name: str, ns: Optional[bool] = None):
        self.name: str = name
        if not self.name in __CLOCKS__:
            __create_clock__(name, ns or False)
        self.ns = __CLOCKS__[name].in_nanoseconds

    def __enter__(self):
        self.start_time = time.perf_counter() if not self.ns else time.perf_counter_ns()

    def elapsed_time(self) -> float:
        return (
            time.perf_counter() if not self.ns else time.perf_counter_ns()
        ) - self.start_time

    def __exit__(self, exc_type, exc_value, exc_traceback):
        elapsed_time = self.elapsed_time()
        __add_entry_to_clock__(self.name, elapsed_time)


def clock(name: str, ns: Optional[bool] = None) -> ClockContextManager:
    return ClockContextManager(name, ns)


def clockfn(
    _func=None, *, prefix: Optional[str] = None, nanoseconds: Optional[bool] = None
):
    def decorator(func: Callable) -> Callable:
        local_name = func.__name__
        if prefix:
            local_name = prefix + "." + local_name

        @wraps(func)
        def wrapper_func(*args, **kwargs):
            with clock(local_name, nanoseconds):
                return func(*args, **kwargs)

        return wrapper_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)
