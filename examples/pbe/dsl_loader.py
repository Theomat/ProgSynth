from argparse import ArgumentParser
import importlib
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional


def __base_loader(
    name: str, keys: Iterable[str] = ["dsl", "evaluator", "lexicon"]
) -> Callable[[bool], Optional[SimpleNamespace]]:
    def loader(fully_load: bool = True) -> Optional[SimpleNamespace]:
        if not fully_load:
            return importlib.util.find_spec(name + "." + name)

        module = importlib.import_module(name + "." + name)
        out = {}
        for key in keys:
            out[key] = module.__getattribute__(key)
        return SimpleNamespace(**out)

    return loader


__dsl_funcs: Dict[str, Callable[[bool], Optional[SimpleNamespace]]] = {
    "deepcoder": __base_loader("deepcoder"),
    "dreamcoder": __base_loader("deepcoder"),
    "regexp": __base_loader(
        "regexp",
        ["dsl", "evaluator", "lexicon", "pretty_print_inputs", "pretty_print_inputs"],
    ),
    "transduction": __base_loader(
        "transduction", ["dsl", "evaluator", "lexicon", "constant_types"]
    ),
    "calculator": __base_loader("calculator"),
}
__buffer: Dict[str, Optional[SimpleNamespace]] = {}


def __wrapper_load(dsl_name: str, fully_load: bool = True) -> Optional[SimpleNamespace]:
    if dsl_name in __buffer:
        return __buffer[dsl_name]
    out = __dsl_funcs[dsl_name](fully_load)
    if fully_load:
        __buffer[dsl_name] = out
    return out


def available_DSL() -> List[str]:
    return [
        dsl_name
        for dsl_name in __dsl_funcs
        if __wrapper_load(dsl_name, False) is not None
    ]


def load_DSL(name: str) -> SimpleNamespace:
    if name not in __dsl_funcs:
        raise Exception(f"DSL {name} is not known!")
    return __wrapper_load(name, True)


def add_dsl_choice_arg(parser: ArgumentParser) -> None:
    dsls = available_DSL()
    parser.add_argument(
        "--dsl",
        type=str,
        default=dsls[0],
        choices=dsls,
        help=f"DSL choice (default: {dsls[0]})",
    )