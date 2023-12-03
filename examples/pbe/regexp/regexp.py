import re

from examples.pbe.regexp.type_regex import regex_match, Raw, REGEXP
from examples.pbe.regexp.evaluator_regexp import RegexpEvaluator, get_regexp
from examples.pbe.regexp.task_generator_regexp import reproduce_regexp_dataset

from synth.semantic import DSLEvaluator
from synth.syntax import DSL, PrimitiveType, Arrow, List, STRING, BOOL


def pretty_print_solution(regexp: str) -> str:
    result = (
        "".join("".join(regexp.__str__().split("(")[2:]).split(" ")[::-1])
        .replace(")", "")
        .replace("begin", "")
    )
    return f"(eval var0 {result})"


def pretty_print_inputs(str: List) -> str:
    return "'" + "".join(str) + "'"


init = PrimitiveType("")


def __qmark__(x):
    return x + "?"


def __kleene__(x):
    return x + "*"


def __plus__(x):
    return x + "+"


def __lowercase__(x):
    return x + "L"


def __uppercase__(x):
    return x + "U"


def __number__(x):
    return x + "N"


def __other__(x):
    return x + "O"


def __whitespace__(x):
    return x + "W"


def __eval__(x, reg):
    x = "".join(x)
    result = regex_match(Raw(get_regexp(reg)), x, flags=re.ASCII)
    # print(f"{result.match.group() if result else None} vs {x} => {result.string == x if result != None else False}")
    if result is None:
        return False
    return result.match.group() == x


__semantics = {
    "begin": init.type_name,
    "?": __qmark__,
    "*": __kleene__,
    "+": __plus__,
    "U": __uppercase__,
    "L": __lowercase__,
    "N": __number__,
    "O": __other__,
    "W": __whitespace__,
    "eval": lambda x: lambda reg: __eval__(x, reg),
}

__primitive_types = {
    "begin": REGEXP,
    "?": Arrow(REGEXP, REGEXP),
    "*": Arrow(REGEXP, REGEXP),
    "+": Arrow(REGEXP, REGEXP),
    "U": Arrow(REGEXP, REGEXP),
    "L": Arrow(REGEXP, REGEXP),
    "N": Arrow(REGEXP, REGEXP),
    "O": Arrow(REGEXP, REGEXP),
    "W": Arrow(REGEXP, REGEXP),
    "eval": Arrow(List(STRING), Arrow(REGEXP, BOOL)),
}

__forbidden_patterns = {
    "*": {"?", "+", "*"},
    "?": {"?", "+", "*"},
    "+": {"?", "+", "*"},
    "W": {"?", "+", "*"},
}

dsl = DSL(__primitive_types, __forbidden_patterns)
evaluator = DSLEvaluator(dsl.instantiate_semantics(__semantics))
evaluator.skip_exceptions.add(re.error)
lexicon = list([chr(i) for i in range(32, 126)])
regexp_evaluator = RegexpEvaluator(__semantics)
