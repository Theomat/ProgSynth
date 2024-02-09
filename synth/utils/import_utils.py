import importlib
from types import SimpleNamespace
from typing import List, Tuple, TypeVar, Union, Iterable, Callable, Optional

U = TypeVar("U")


def __try_names__(name: str, f: Callable[[str], U], prefixes: List[str]) -> U:
    try:
        return f(name)
    except ModuleNotFoundError:
        l = prefixes[0]
        return __try_names__(l + "." + name, f, prefixes[1:])


def import_file_function(
    import_name: str,
    keys: Iterable[Union[str, Tuple[str, str]]],
    prefixes: List[str] = [],
) -> Callable[[bool], Optional[SimpleNamespace]]:
    """
    Utility function that creates a simple loader for you if you only need
    > from name.name import X, Y, ...
    where "X, Y, ..." are elements of keys.

    prefixes is a list of prefixes to the import name to try in case of failure.
    """

    def loader(fully_load: bool = True) -> Optional[SimpleNamespace]:
        if not fully_load:
            return __try_names__(import_name, importlib.util.find_spec, prefixes)  # type: ignore

        module = __try_names__(import_name, importlib.import_module, prefixes)
        out = {}
        for key in keys:
            get, set = key if isinstance(key, tuple) else (key, key)
            out[set] = module.__getattribute__(get)
        return SimpleNamespace(**out)

    return loader
