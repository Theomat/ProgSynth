from dataclasses import dataclass
from typing import (
    Tuple,
    List as TList,
)
from synth.syntax.grammars.det_grammar import DerivableProgram


from synth.syntax.grammars.ttcfg import TTCFG
from synth.syntax.program import Variable

# ========================================================================================
# SYMBOLS
# ========================================================================================
SYMBOL_ANYTHING = "_"
SYMBOL_VAR_EXPR = "$"
SYMBOL_FORBIDDEN = "^"
SYMBOL_SEPARATOR = ","
SYMBOL_AGGREGATOR = "#"
# ========================================================================================
# TOKENS
# ========================================================================================
class Token:
    def __repr__(self) -> str:
        return str(self)


class TokenAnything(Token):
    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "Any"


@dataclass
class TokenAllow(Token):
    allowed: TList[DerivableProgram]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Allow ({self.allowed})"


@dataclass
class TokenVarDep(Token):
    variables: TList[Variable]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"VarDep ({self.variables})"


@dataclass
class TokenAtMost(Token):
    to_count: TList[DerivableProgram]
    count: int

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"AtMost {self.count} ({self.to_count})"


@dataclass
class TokenAtLeast(Token):
    to_count: TList[DerivableProgram]
    count: int

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"AtLeast {self.count} ({self.to_count})"


@dataclass
class TokenFunction(Token):
    function: TokenAllow
    args: TList[Token]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Func f={self.function} args=({self.args})"


# ========================================================================================
# PARSING
# ========================================================================================
def __next_level__(string: str, start: str, end: str) -> int:
    level = 0
    for i, el in enumerate(string):
        if el == start:
            level += 1
        if el == end:
            level -= 1
            if level == 0:
                return i
    return i


def __parse_next_word__(program: str) -> Tuple[str, int]:
    if program[0] in [SYMBOL_VAR_EXPR, "("]:
        end = __next_level__(program, "(", ")")
    else:
        end = program.index(" ") - 1 if " " in program else len(program) - 1
    return program[: end + 1], end + 2


def __str_to_derivable_program__(word: str, grammar: TTCFG) -> TList[DerivableProgram]:
    allowed = set(
        [word] if not SYMBOL_SEPARATOR in word else word.split(SYMBOL_SEPARATOR)
    )
    primitives: TList[DerivableProgram] = [
        P for P in grammar.primitives_used() if P.primitive in allowed
    ]
    arg_types = grammar.type_request.arguments()
    for el in allowed:
        if el.startswith("var"):
            varno = int(el[3:])
            primitives.append(Variable(varno, arg_types[varno]))
    return primitives


def __interpret_word__(word: str, grammar: TTCFG) -> Token:
    word = word.strip()
    if word.startswith(SYMBOL_FORBIDDEN):
        forbidden = set(word[1:].split(SYMBOL_SEPARATOR))
        # TODO: Add variables
        return TokenAllow(
            [P for P in grammar.primitives_used() if P.primitive not in forbidden]
        )
    elif word.startswith(SYMBOL_VAR_EXPR):
        var_text = word.strip("()" + SYMBOL_VAR_EXPR)
        return TokenVarDep(__str_to_derivable_program__(var_text, grammar))  # type: ignore
    elif word == SYMBOL_ANYTHING:
        return TokenAnything()
    elif word.startswith(SYMBOL_AGGREGATOR):
        word = word[1:].replace(" ", "")
        end_index = max(word.find("<="), word.find(">="))
        most = word[end_index] == "<"
        considered = word[:end_index]
        count = int(word[len(considered) + 2 :])
        if most:
            return TokenAtMost(__str_to_derivable_program__(considered, grammar), count)
        else:
            return TokenAtLeast(
                __str_to_derivable_program__(considered, grammar), count
            )
    return TokenAllow(__str_to_derivable_program__(word, grammar))


def parse_specification(spec: str, grammar: TTCFG) -> Token:
    spec = spec.replace("\n", "").strip(")(")
    index = 0
    elements = []
    while index < len(spec):
        spec = spec[index:]
        word, index = __parse_next_word__(spec)
        if word.startswith("("):
            token = parse_specification(word, grammar)
        else:
            token = __interpret_word__(word, grammar)
        elements.append(token)
    assert len(elements) > 0
    first = elements.pop(0)
    assert isinstance(first, TokenAllow)
    return TokenFunction(first, elements)
