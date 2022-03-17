from ctypes.wintypes import PBYTE
import sys
from examples.pbe.regexp.evaluator_regexp import get_regexp
from synth.specification import PBE

from synth.task import Dataset
from synth.semantic import DSLEvaluator
from synth.syntax import DSL, PrimitiveType, Arrow, List, INT, STRING


import string
import re
from examples.pbe.regexp.type_regex import regex_match, Raw, REGEXP, regex_search

from synth.syntax.type_system import BOOL, PolymorphicType

STREGEXP = PolymorphicType("str/regexp")


def pretty_print_solution(regexp: str) -> str:
    result = (
        "".join("".join(regexp.__str__().split("(")[2:]).split(" ")[::-1])
        .replace(")", "")
        .replace("begin", "")
    )
    return f"(eval var0 {result})"


def pretty_print_inputs(str: List) -> str:
    return "'" + "".join(str) + "'"


def __concat__(x, y):
    return "" + x + y


def __lower__(x: str):
    return x.lower()


def __upper__(x: str):
    return x.upper()

def __split__(x: str, regexp: str):
    sbstr = regex_search(Raw(get_regexp(regexp)), x, flags=re.ASCII)
    if sbstr == None:
        return ["", ""]
    return x.split(sbstr.match.group(), 1)

def __head__(x):
    return x[0]

def __tail__(x):
    return x[1]


__semantics = {
    "concat": lambda x: lambda y: __concat__(x, y),
    "lower": __lower__,
    "upper": __upper__,
    "split": lambda x: lambda regexp: __split__(x, regexp),
    "head": __head__,
    "tail": __tail__,
    "O": "O",
    "W": " ",
}

__primitive_types = {
    "concat": Arrow(STRING, Arrow(STREGEXP, STRING)),
    "lower": Arrow(STRING, STRING),
    "upper": Arrow(STRING, STRING),
    "split": Arrow(STRING, Arrow(REGEXP, List(STRING))),
    "head": Arrow(List(STRING), STRING),
    "tail": Arrow(List(STRING), STRING),
    "O": REGEXP,
    "W": REGEXP,
}

__forbidden_patterns = []

dsl = DSL(__primitive_types, __forbidden_patterns)
evaluator = DSLEvaluator(__semantics)
evaluator.skip_exceptions.add(re.error)
lexicon = list([chr(i) for i in range(32, 126)])
