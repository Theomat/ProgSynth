from typing import Set
import re

from examples.pbe.transduction.task_generator_transduction import (
    reproduce_transduction_dataset,
)

from synth.semantic.evaluator import DSLEvaluatorWithConstant
from synth.syntax import (
    DSL,
    Arrow,
    STRING,
    PrimitiveType,
)
from examples.pbe.regexp.evaluator_regexp import get_regexp
from examples.pbe.regexp.type_regex import REGEXP
from examples.pbe.regexp.type_regex import (
    Raw,
    REGEXP,
    regex_search,
)

CST_IN = PrimitiveType("CST_STR_INPUT")
CST_OUT = PrimitiveType("CST_STR_OUTPUT")


def __concat__(x, y):
    return "" + x + y


def __concat_if__(x, y):
    if y in x:
        return x
    if x in y:
        return y
    return "" + x + y


def __split_first__(x: str, regexp: str):
    sbstr = regex_search(Raw(get_regexp(regexp)), x, flags=re.ASCII)
    if sbstr == None:
        return ""
    return x.split(sbstr.match.group(), 1)[0]


def __split_snd__(x: str, regexp: str):
    sbstr = regex_search(Raw(get_regexp(regexp)), x, flags=re.ASCII)
    if sbstr == None:
        return ""
    return x.split(sbstr.match.group(), 1)[1]


def __match__(x: str, regexp: str):
    sbstr = regex_search(Raw(get_regexp(regexp)), x, flags=re.ASCII)
    if sbstr == None:
        return ""
    return sbstr.match.group()


# untreated matching, done for constant text inputs (e.g. "." will be considered as a point instead of any char)
def __split_first_cst__(x: str, text: str):
    regexp = "(\\" + text + ")"
    sbstr = regex_search(Raw(regexp), x, flags=re.ASCII)
    if sbstr == None:
        return ""
    return x.split(sbstr.match.group(), 1)[0]


def __split_snd_cst__(x: str, text: str):
    regexp = "(\\" + text + ")"
    sbstr = regex_search(Raw(regexp), x, flags=re.ASCII)
    if sbstr == None:
        return ""
    return x.split(sbstr.match.group(), 1)[1]


def __match_cst__(x: str, text: str):
    regexp = "(\\" + text + ")"
    sbstr = regex_search(Raw(regexp), x, flags=re.ASCII)
    if sbstr == None:
        return ""
    return sbstr.match.group()


def __compose__(x, y):
    return x + y


__semantics = {
    "concat": lambda x: lambda y: __concat__(x, y),
    "concat_cst": lambda x: lambda y: __concat__(x, y),
    "concat_if": lambda x: lambda y: __concat_if__(x, y),
    "split_first": lambda x: lambda regexp: __split_first__(x, regexp),
    "split_snd": lambda x: lambda regexp: __split_snd__(x, regexp),
    "match": lambda x: lambda regexp: __match__(x, regexp),
    "split_first_cst": lambda x: lambda text: __split_first_cst__(x, text),
    "split_snd_cst": lambda x: lambda text: __split_snd_cst__(x, text),
    "match_cst": lambda x: lambda text: __match_cst__(x, text),
    "compose": lambda x: lambda y: __compose__(x, y),
    "cst_in": lambda x: x,
    "cst_out": lambda x: x,
    "$": "$",
    ".": ".",
    "except": lambda x: "([^" + x + "]+",
    "except_end": lambda x: "([^" + x + "]+$",
}

__primitive_types = {
    "concat": Arrow(STRING, Arrow(STRING, STRING)),
    "concat_cst": Arrow(STRING, Arrow(CST_OUT, STRING)),
    "concat_if": Arrow(STRING, Arrow(CST_OUT, STRING)),
    "split_first": Arrow(STRING, Arrow(REGEXP, STRING)),
    "split_snd": Arrow(STRING, Arrow(REGEXP, STRING)),
    "match": Arrow(STRING, Arrow(REGEXP, STRING)),
    "split_first_cst": Arrow(STRING, Arrow(CST_IN, STRING)),
    "split_snd_cst": Arrow(STRING, Arrow(CST_IN, STRING)),
    "match_cst": Arrow(STRING, Arrow(CST_IN, STRING)),
    "compose": Arrow(REGEXP, Arrow(REGEXP, REGEXP)),
    "cst_in": CST_IN,
    "cst_out": CST_OUT,
    "$": REGEXP,
    ".": REGEXP,
    "except": Arrow(CST_IN, REGEXP),
    "except_end": Arrow(CST_IN, REGEXP),
}

__forbidden_patterns = {}

dsl = DSL(__primitive_types, __forbidden_patterns)
constant_types = {CST_IN, CST_OUT}
evaluator = DSLEvaluatorWithConstant(__semantics, constant_types)
evaluator.skip_exceptions.add(re.error)
lexicon = list([chr(i) for i in range(32, 126)])
constraints = [
    "concat ^concat _",
    "compose ^compose _",
]
