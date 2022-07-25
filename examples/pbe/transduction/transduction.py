from typing import (
    Set,
    List as TList,
    Any,
    Optional,
    Tuple,
)
import re

from synth.pbe.task_generator import (
    basic_output_validator,
    reproduce_dataset,
    TaskGenerator,
)
from synth.semantic.evaluator import DSLEvaluatorWithConstant
from synth.task import Dataset
from synth.specification import PBE
from synth.semantic import Evaluator
from synth.syntax import (
    DSL,
    Arrow,
    STRING,
    PrimitiveType,
)
from synth.generation.sampler import (
    LexiconSampler,
    UnionSampler,
)

from examples.pbe.regexp.evaluator_regexp import get_regexp
from examples.pbe.regexp.type_regex import REGEXP
from examples.pbe.regexp.type_regex import (
    Raw,
    REGEXP,
    regex_search,
)

CSTE_IN = PrimitiveType("CST_STR_INPUT")
CSTE_OUT = PrimitiveType("CST_STR_OUTPUT")


def __concat__(x, y):
    return "" + x + y


def __concat_if__(x, y):
    if y in x:
        return x
    if x in y:
        return y
    return "" + x + y


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


# untreated matching, done for constant text inputs (e.g. "." will be considered as a point instead of any char)
def __head_text__(x: str, text: str):
    regexp = "(\\" + text + ")"
    sbstr = regex_search(Raw(regexp), x, flags=re.ASCII)
    if sbstr == None:
        return ""
    return x.split(sbstr.match.group(), 1)[0]


def __tail_text__(x: str, text: str):
    regexp = "(\\" + text + ")"
    sbstr = regex_search(Raw(regexp), x, flags=re.ASCII)
    if sbstr == None:
        return ""
    return x.split(sbstr.match.group(), 1)[1]


def __match_text__(x: str, text: str):
    regexp = "(\\" + text + ")"
    sbstr = regex_search(Raw(regexp), x, flags=re.ASCII)
    if sbstr == None:
        return ""
    return sbstr.match.group()


def __compose__(x, y):
    return x + y


__semantics = {
    "concat": lambda x: lambda y: __concat__(x, y),
    "concat_cste": lambda x: lambda y: __concat__(x, y),
    "concat_if": lambda x: lambda y: __concat_if__(x, y),
    "head": lambda x: lambda regexp: __head__(x, regexp),
    "tail": lambda x: lambda regexp: __tail__(x, regexp),
    "match": lambda x: lambda regexp: __match__(x, regexp),
    "head_cste": lambda x: lambda text: __head_text__(x, text),
    "tail_cste": lambda x: lambda text: __tail_text__(x, text),
    "match_cste": lambda x: lambda text: __match_text__(x, text),
    "compose": lambda x: lambda y: __compose__(x, y),
    "cste_in": lambda x: x,
    "cste_out": lambda x: x,
    "$": "$",
    ".": ".",
    "except": lambda x: "([^" + x + "]+",
    "except_end": lambda x: "([^" + x + "]+$",
}

__primitive_types = {
    "concat": Arrow(STRING, Arrow(STRING, STRING)),
    "concat_cste": Arrow(STRING, Arrow(CSTE_OUT, STRING)),
    "concat_if": Arrow(STRING, Arrow(CSTE_OUT, STRING)),
    "head": Arrow(STRING, Arrow(REGEXP, STRING)),
    "tail": Arrow(STRING, Arrow(REGEXP, STRING)),
    "match": Arrow(STRING, Arrow(REGEXP, STRING)),
    "head_cste": Arrow(STRING, Arrow(CSTE_IN, STRING)),
    "tail_cste": Arrow(STRING, Arrow(CSTE_IN, STRING)),
    "match_cste": Arrow(STRING, Arrow(CSTE_IN, STRING)),
    "compose": Arrow(REGEXP, Arrow(REGEXP, REGEXP)),
    "cste_in": CSTE_IN,
    "cste_out": CSTE_OUT,
    "$": REGEXP,
    ".": REGEXP,
    "except": Arrow(CSTE_IN, REGEXP),
    "except_end": Arrow(CSTE_IN, REGEXP),
}

__forbidden_patterns = {}

dsl = DSL(__primitive_types, __forbidden_patterns)
constant_types: Set[PrimitiveType] = set()
constant_types.add(CSTE_IN)
constant_types.add(CSTE_OUT)
evaluator = DSLEvaluatorWithConstant(__semantics, constant_types)
evaluator.skip_exceptions.add(re.error)
lexicon = list([chr(i) for i in range(32, 126)])


def reproduce_transduction_dataset(
    dataset: Dataset[PBE],
    dsl: DSL,
    evaluator: Evaluator,
    seed: Optional[int] = None,
    *args: Any,
    **kwargs: Any
) -> Tuple[TaskGenerator, TList[int]]:
    def analyser(start: None, elment: Any) -> None:
        pass

    str_lexicon = list([chr(i) for i in range(32, 126)])

    def get_sampler(start: None) -> UnionSampler:
        return UnionSampler(
            {
                STRING: LexiconSampler(str_lexicon, seed=seed),
                REGEXP: LexiconSampler(
                    [
                        "_",
                        ")",
                        "{",
                        "+",
                        ";",
                        "=",
                        "$",
                        "\\",
                        "^",
                        ",",
                        "!",
                        "*",
                        "'",
                        " ",
                        ">",
                        "}",
                        "<",
                        "[",
                        '"',
                        "#",
                        "|",
                        "`",
                        "%",
                        "?",
                        ":",
                        "]",
                        "&",
                        "(",
                        "@",
                        ".",
                        "/",
                        "-",
                    ],
                    seed=seed,
                ),
            }
        )

    return reproduce_dataset(
        dataset,
        dsl,
        evaluator,
        None,
        lambda _, __: None,
        get_sampler,
        lambda _, max_list_length: basic_output_validator(str_lexicon, max_list_length),
        lambda _: str_lexicon,
        seed,
        **args,
        **kwargs
    )
