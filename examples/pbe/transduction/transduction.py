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

CSTE = PrimitiveType("STR")

def __concat__(x, y):
    return "" + x + y

def __concat_space__(x, y):
    return "" + x + " " + y

def __separate__(x: str, regexp: str):
    match = regex_findall(Raw(get_regexp(regexp, False)), x, flags=re.ASCII)
    if match == None:
        return ""
    return ' '.join(match.match)

def __head__(x: str, regexp: str):
    sbstr = regex_search(Raw(get_regexp(regexp)), x, flags=re.ASCII)
    if sbstr == None:
        return ""
    return x.split(sbstr.match.group(), 1)[0]

def __tail__(x: str, regexp: str):
    sbstr = regex_search(Raw(get_regexp(regexp)), x, flags=re.ASCII)
    if sbstr == None:
        return ""
    return x.split(sbstr.match.group(), 1)[1]

def __match__(x: str, regexp: str):
    sbstr = regex_search(Raw(get_regexp(regexp)), x, flags=re.ASCII)
    if sbstr == None:
        return ""
    return sbstr.match.group()

def __compose__(x, y):
    return x + y

__semantics = {
    "concat": lambda x: lambda y: __concat__(x, y),
    #"concat_space": lambda x: lambda y: __concat_space__(x, y),
    "concat_cste": lambda x: lambda y: __concat__(x, y),
    "head": lambda x: lambda regexp: __head__(x, regexp), 
    "tail": lambda x: lambda regexp: __tail__(x, regexp), 
    "match": lambda x: lambda regexp: __match__(x, regexp), 
    "separate": lambda x: lambda regexp: __separate__(x, regexp),
    "compose": lambda x: lambda y: __compose__(x, y),
    #"compose+": lambda x: __compose__(x, "+"),
    "cste": lambda x: x,
    "U": "U",
    "L": "L",
    "N": "N",
    "O": "O",
    "W": "W",
    "$": "$",
    ".": ".",
}

__primitive_types = {
    "concat": Arrow(STRING, Arrow(STRING, STRING)),
    #"concat_space": Arrow(STRING, Arrow(STRING, STRING)),
    "concat_cste": Arrow(STRING, Arrow(CSTE, STRING)),
    "head": Arrow(STRING, Arrow(REGEXP, STRING)),
    "tail": Arrow(STRING, Arrow(REGEXP, STRING)),
    "match": Arrow(STRING, Arrow(REGEXP, STRING)),
    "separate": Arrow(STRING, Arrow(REGEXP, STRING)),
    "compose": Arrow(REGEXP, Arrow(REGEXP, REGEXP)),
    #"compose+": Arrow(REGEXP, REGEXP),
    "cste": CSTE,
    "U": REGEXP,
    "L": REGEXP,
    "N": REGEXP,
    "O": REGEXP,
    "W": REGEXP,
    "$": REGEXP,
    ".": REGEXP
}

__forbidden_patterns = [
]

dsl = DSL(__primitive_types, __forbidden_patterns)
evaluator = DSLEvaluator(__semantics)
evaluator.skip_exceptions.add(re.error)
lexicon = list([chr(i) for i in range(32, 126)])
constant_types = set()
constant_types.add(CSTE)