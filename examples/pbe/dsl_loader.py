"""
Module to change to add your own DSL easily in all scripts.
Some constants may need to be chnaged directly in the script. 
"""
from argparse import ArgumentParser
import importlib
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union


def __base_loader(
    name: str,
    keys: Iterable[Union[str, Tuple[str, str]]] = ["dsl", "evaluator", "lexicon"],
) -> Callable[[bool], Optional[SimpleNamespace]]:
    """
    Utility function that creates a simple loader for you if you only need
    > from name.name import X, Y, ...
    where "X, Y, ..." are elements of keys
    """

    def loader(fully_load: bool = True) -> Optional[SimpleNamespace]:
        if not fully_load:
            return importlib.util.find_spec(name)

        module = importlib.import_module(name)
        out = {}
        for key in keys:
            get, set = key if isinstance(key, tuple) else (key, key)
            out[set] = module.__getattribute__(get)
        return SimpleNamespace(**out)

    return loader


# ======================================================================================
# ADD your line "my_dsl": my_loading_func
#   when the parameter to your loading_func is:
#       - False, we just want to check we CAN import
#       - True, we want to import everything and return it in a NameSpace
# /!\ Convention for names in your namespace:
#   dsl: DSL - the actual DSL
#   evaluator: Evaluator - the DSL's evaluator
#   lexicon: List - the DSL's lexicon
# =======================================================================================
__dsl_funcs: Dict[str, Callable[[bool], Optional[SimpleNamespace]]] = {
    "deepcoder": __base_loader("deepcoder.deepcoder"),
    "deepcoder.raw": __base_loader(
        "deepcoder.deepcoder", [("dsl_raw", "dsl"), "evaluator", "lexicon"]
    ),
    "deepcoder.pruned": __base_loader("deepcoder.deepcoder_pruned"),
    "dreamcoder": __base_loader("dreamcoder.dreamcoder"),
    "regexp": __base_loader(
        "regexp.regexp",
        ["dsl", "evaluator", "lexicon", "pretty_print_inputs", "pretty_print_inputs"],
    ),
    "transduction": __base_loader(
        "transduction.transduction", ["dsl", "evaluator", "lexicon", "constant_types"]
    ),
    "calculator": __base_loader("calculator.calculator"),
}
# =======================================================================================
# Nothing to change after this
# =======================================================================================

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
