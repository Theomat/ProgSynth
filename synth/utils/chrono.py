"""
Module to measure time spent in parts of the code programmatically and easily.
"""
from functools import wraps
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Union
from colorama import init, Fore

init()


@dataclass
class ClockData:
    total: float = field(default=0)
    count: int = field(default=0)
    max: float = field(default=0)
    min: float = field(default=0)
    mean: float = field(default=0)
    _square_sum: float = field(default=0)
    autofilled: bool = field(default=True)

    def add_data(self, time: float) -> None:
        self.autofilled = False
        if self.count == 0:
            self.min = time
            self.max = time
        else:
            self.min = min(self.min, time)
            self.max = max(self.max, time)
        self.count += 1
        self.total += time
        delta = time - self.mean
        self.mean += delta / self.count
        delta2 = time - self.mean
        self._square_sum += delta * delta2

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return float("nan")
        return self._square_sum / (self.count - 1)

    def __str__(self) -> str:
        return f"total={self.total}s range=[{self.min}-{self.max}] mean={self.mean}~{self.variance}"


@dataclass(frozen=True)
class PrefixTree:
    name: str
    data: ClockData = field(default_factory=lambda: ClockData(), compare=False)
    children: List["PrefixTree"] = field(default_factory=lambda: [], compare=False)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_child(self, prefix: str) -> Optional["PrefixTree"]:
        possibles = [child for child in self.children if child.name == prefix]
        return possibles[0] if possibles else None

    def get(self, name: str) -> Tuple["PrefixTree", str]:
        parts = name.split(".")
        current: "PrefixTree" = self
        next = current.get_child(parts[0])
        while next:
            current = next
            parts.pop(0)
            if len(parts) == 0:
                break
            next = current.get_child(parts[0])
        return current, parts  # type: ignore

    def autofill(self) -> None:
        for child in self.children:
            child.autofill()
        if self.data.autofilled:
            self.data.total = sum(child.data.total for child in self.children)
            self.data.count = sum(child.data.count for child in self.children)
            self.data.max = sum(child.data.max for child in self.children)
            self.data.min = sum(child.data.min for child in self.children)
            self.data.mean = sum(child.data.mean for child in self.children)
            self.data._square_sum = sum(
                child.data._square_sum for child in self.children
            )

    def to_string(
        self, time_formatter: Callable[[float], str], tabs: int = 0, colors: bool = True
    ) -> str:
        indent = "\t" * tabs
        # Color management
        light_green = Fore.LIGHTGREEN_EX if colors else ""
        light_yellow = Fore.LIGHTYELLOW_EX if colors else ""
        reset = Fore.RESET if colors else ""

        me = f"{light_green}{self.name}{reset}:" if tabs == 0 else ""
        # Write ClockData
        value = f"total={time_formatter(self.data.total)}"
        value += (
            f" range=[{time_formatter(self.data.min)}-{time_formatter(self.data.max)}]"
        )
        value += f" mean={time_formatter(self.data.mean)}~{time_formatter(self.data.variance)}"

        s = f"{indent}{me} {value}\n"
        for child in self.children:
            time_percent = child.data.total / max(1e-13, self.data.total) * 100
            s += f"{indent}\t- {light_green}{child.name}{reset} ({light_yellow}{time_percent:.2f}%{reset})\n"
            s += child.to_string(time_formatter, tabs + 1, colors)
        return s

    def __str__(self) -> str:
        return self.to_string(lambda t: str(t) + "s")


__ROOT__ = PrefixTree("")


def __node_from_name__(name: str) -> PrefixTree:
    tree, remaining = __ROOT__.get(name)
    for el in remaining:
        if len(el) == 0:
            continue
        leaf = PrefixTree(el)
        tree.children.append(leaf)
        tree = leaf
    return tree


def get(name: str) -> ClockData:
    """
    Get the clock data associated with the specified clock name or empty data if such a clock does not exist.
    """
    return __node_from_name__(name).data


def summary(
    time_formatter: Callable[[float], str], domain: str = "", colors: bool = True
) -> str:
    # Build the tree
    root = __node_from_name__(domain)
    root.autofill()
    return root.to_string(time_formatter, colors=colors)


class ClockContextManager:
    def __init__(self, name: str):
        self.data = get(name)

    def __enter__(self) -> "ClockContextManager":
        self.start_time = time.perf_counter()
        return self

    def elapsed_time(self) -> float:
        """
        Returned elapsed time since this context was opened in the unit used by the clock.
        """
        return time.perf_counter() - self.start_time

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        elapsed_time = self.elapsed_time()
        self.data.add_data(elapsed_time)


def clock(
    name: Union[str, Callable] = "", *, prefix: Optional[str] = None
) -> Union[ClockContextManager, Callable]:
    """
    Measure time taken either as a context manager or a function decorator.

    As a context manager:
        - name (str) - name of the clock
        - nanoseconds (bool) - whether to record time in nanoseconds. Default to False
    As a function decorator:
        - prefix (str) - prefix to the function name to add for the name of the clock
        - nanoseconds (bool) - whether to record time in nanoseconds. Default to False

    """
    if isinstance(name, str) and len(name) > 0:
        return ClockContextManager(name)
    _func = name

    def decorator(func: Callable) -> Callable:
        local_name = func.__name__
        if prefix:
            local_name = prefix + "." + local_name
        clock_data = get(local_name)

        @wraps(func)
        def wrapper_func(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            out = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            clock_data.add_data(elapsed_time)
            return out

        return wrapper_func

    if isinstance(_func, str):
        return decorator
    else:
        return decorator(_func)
