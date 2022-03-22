from ctypes.wintypes import PBYTE
import sys
from examples.pbe.regexp.evaluator_regexp import get_regexp
from synth.specification import PBE

from synth.task import Dataset
from synth.semantic import DSLEvaluator
from synth.syntax import DSL, PrimitiveType, Arrow, List, INT, STRING


import string
import re
from examples.pbe.regexp.type_regex import regex_findall, regex_match, Raw, REGEXP, regex_search

from synth.syntax.type_system import BOOL, PolymorphicType

STREGEXP = PolymorphicType("str/regexp")

def __concat__(x, y):
    return "" + x + " " + y


def __lower__(x: str):
    return x.lower()


def __upper__(x: str):
    return x.upper()

def __split__(x: str, regexp: str):
    sbstr = regex_search(Raw(get_regexp(regexp)), x, flags=re.ASCII)
    if sbstr == None:
        return ["", "", ""]
    return x.split(sbstr.match.group(), 1)

def __separate__(x: str, regexp: str):
    match = regex_findall(Raw(get_regexp(regexp, False)), x, flags=re.ASCII)
    if match == None:
        return ""
    return ' '.join(match.match)

def __head__(x):
    return x[0]

def __tail__(x):
    return x[1]

def __compose__(x, y):
    return x + y

__semantics = {
    "concat": lambda x: lambda y: __concat__(x, y),
    "lower": __lower__,
    "upper": __upper__,
    "split": lambda x: lambda regexp: __split__(x, regexp),
    "head": __head__,
    "tail": __tail__,
    "separate": lambda x: lambda regexp: __separate__(x, regexp),
    "compose": lambda x: lambda y: __compose__(x, y),
    "U": "U",
    "L": "L",
    "N": "N",
    "O": "O",
    "W": "W",
}

__primitive_types = {
    "concat": Arrow(STRING, Arrow(STRING, STRING)),
    "lower": Arrow(STRING, STRING),
    "upper": Arrow(STRING, STRING),
    "split": Arrow(STRING, Arrow(REGEXP, List(STRING))),
    "head": Arrow(List(STRING), STRING),
    "tail": Arrow(List(STRING), STRING),
    "separate": Arrow(STRING, Arrow(REGEXP, STRING)),
    "compose": Arrow(REGEXP, Arrow(REGEXP, REGEXP)),
    "U": REGEXP,
    "L": REGEXP,
    "N": REGEXP,
    "O": REGEXP,
    "W": REGEXP,
}

__forbidden_patterns = []

dsl = DSL(__primitive_types, __forbidden_patterns)
evaluator = DSLEvaluator(__semantics)
evaluator.skip_exceptions.add(re.error)
lexicon = list([chr(i) for i in range(32, 126)])
