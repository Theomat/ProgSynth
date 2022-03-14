from ctypes.wintypes import PBYTE
import sys
from synth.specification import PBE

from synth.task import Dataset
from synth.semantic import DSLEvaluator
from synth.syntax import DSL, PrimitiveType, Arrow, List, INT, STRING


import string
import re
from examples.pbe.regexp.type_regex import regex_match, Raw, REGEXP
from examples.pbe.regexp.evaluator_regexp import RegexpEvaluator, get_regexp


from synth.syntax.type_system import BOOL


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

def __concat__(x, y):
    return '' + x + y

def __lower__(x: str):
    return x.lower

def __upper__(x: str):
    return x.upper

def __plus_operator__(x):
    return x+1

"""
either this (poor design in my opinion, not using regexp), or with regexp. 
The second case requires a very deep depth program however
"""
def __substr__(x, i, j):
    return x[i:j]

def __split__(x: str, chr: str, i: int):
    return x.split(chr)[i]

"""
def __alt__(x, y): pass # x|y (one of the two)

def __min__(x, mi): pass #x{mi,} (repeated at least mi times)

def __max__(x, ma): pass #x{,ma} (repeated up to ma times)

def __borned__(x, mi, ma): pass #x{mi,ma} (repeated between mi and ma times)
"""
# list to be taken from re (https://docs.python.org/3/library/re.html), used here as reference
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
    #"concat": lambda x: lambda y:__concat__(x, y),
    #"lower": __lower__,
    #"upper": __upper__,
    #"0": 0,
    #"+": __plus_operator__,
    #"sbstr": lambda x: lambda i: lambda j: __substr__(x, i, j),
    #"split": lambda x: lambda chr: lambda i: __split__(x, chr, i),

}

__primitive_types = {
    "begin": REGEXP,
    "0": INT,
    "?": Arrow(REGEXP, REGEXP),
    "*": Arrow(REGEXP, REGEXP),
    "+": Arrow(REGEXP, REGEXP),
    "U": Arrow(REGEXP, REGEXP),
    "L": Arrow(REGEXP, REGEXP),
    "N": Arrow(REGEXP, REGEXP),
    "O": Arrow(REGEXP, REGEXP),
    "W": Arrow(REGEXP, REGEXP),
    "eval": Arrow(List(STRING), Arrow(REGEXP, BOOL)),
    #"concat": Arrow(STRING, Arrow(STRING, STRING)),
    #"lower": Arrow(STRING, STRING),
    #"upper": Arrow(STRING, STRING),
    #"+": Arrow(INT, INT),
    #"sbstr": Arrow(STRING, Arrow(INT, Arrow(INT, STRING))),
    #"split": Arrow(STRING, Arrow(STRING, Arrow(INT, STRING))),
}

__forbidden_patterns = [
    ["*", "?"],
    ["*", "+"],
    ["*", "*"],
    ["?", "?"],
    ["?", "+"],
    ["?", "*"],
    ["+", "?"],
    ["+", "+"],
    ["+", "*"],
    ["W", "?"],
    ["W", "+"],
    ["W", "*"],
]

dsl = DSL(__primitive_types, __forbidden_patterns)
evaluator = DSLEvaluator(__semantics)
evaluator.skip_exceptions.add(re.error)
lexicon = list([chr(i) for i in range(32, 126)])
regexp_evaluator = RegexpEvaluator(__semantics)
