"""
An helper file that contains useful methods to make type creation and manipulation very easy.

"""
from synth.syntax.type_system import FixedPolymorphicType, PrimitiveType
from synth.syntax.type_system import (
    Type,
    UnknownType,
    Arrow,
    Sum,
    EmptyList,
    INT,
    BOOL,
    STRING,
    UNIT,
    List,
    PolymorphicType,
)

from typing import Any, Dict, Tuple, Union, overload


def FunctionType(*args: Type) -> Type:
    """
    Short-hand to create n-ary functions.
    """
    types = list(args)
    base = types.pop()
    while types:
        base = Arrow(types.pop(), base)
    return base


def guess_type(element: Any) -> Type:
    """
    Guess the type of the given element.
    Does not work for Arrow and Polymorphic Types.
    """
    if isinstance(element, (TList, Tuple)):  # type: ignore
        if len(element) == 0:
            return EmptyList
        current: Type = UnknownType()
        i = 0
        while i < len(element) and isinstance(current, UnknownType):
            current = guess_type(element[i])
            i += 1
        return List(current)
    if isinstance(element, bool):
        return BOOL
    elif isinstance(element, int):
        return INT
    elif isinstance(element, str):
        return STRING
    elif element is None:
        return UNIT
    return UnknownType()


_TOK_NONE = -1
_TOK_PARENTHESIS = 0
_TOK_BRACKETS = 1
_TOK_ARROW = 2
_TOK_POLYMORPHIC = 3
_TOK_LIST = 4
_TOK_OR = 5


def __matching__(text: str) -> int:
    level = 0
    start = text[0]
    for j, l in enumerate(text):
        if l == start:
            level += 1
        elif start == "(" and l == ")":
            level -= 1
        elif start == "[" and l == "]":
            level -= 1
        if level == 0:
            return j
    return -1


_SEP_CHARS = ["|", "-", " "]


def __next_token__(text: str) -> Tuple[str, int, int]:
    if text.startswith("(") or text.startswith("["):
        i = __matching__(text)
        return text[1:i], _TOK_BRACKETS if text[0] == "(" else _TOK_PARENTHESIS, i
    elif text.startswith("list") and (len(text) == 4 or text[4] in _SEP_CHARS):
        return "", _TOK_LIST, 4
    elif text.startswith("->"):
        return "", _TOK_ARROW, 2
    elif text.startswith("|"):
        return "", _TOK_OR, 1
    next_index = min(text.index(c) for c in _SEP_CHARS)
    is_poly = len(text) > 0 and text[0] == "'"
    return (
        text[is_poly:next_index],
        _TOK_POLYMORPHIC if is_poly else _TOK_NONE,
        next_index,
    )


@overload
def auto_type(el: str) -> Type:
    pass


@overload
def auto_type(el: Dict[str, str]) -> Dict[str, Type]:
    pass


def auto_type(el: Union[Dict[str, str], str]) -> Union[Dict[str, Type], Type]:
    # Dictionnary part
    if isinstance(el, dict):
        return {k: auto_type(v) for k, v in el.items()}

    # String part
    stack = []
    text = el
    last_arrow = 0
    or_flag = -1
    index = 1
    while index > 0:
        w, token, index = __next_token__(text)
        # index also represents the number of chars consumed
        if token == _TOK_PARENTHESIS:
            stack.append(auto_type(w))
        elif token == _TOK_BRACKETS:
            assert len(stack) > 0
            last = stack.pop()
            assert isinstance(
                last, PolymorphicType
            ), f"Cannot restrain a non polymorphic type:{last}"
            r = auto_type(w)
            stack.append(FixedPolymorphicType(last.name, r))
        elif token == _TOK_LIST:
            assert len(stack) > 0
            stack.append(List(stack.pop()))
        elif token == _TOK_POLYMORPHIC:
            stack.append(PolymorphicType(w))
        elif token == _TOK_NONE:
            stack.append(PrimitiveType(w))
        elif token == _TOK_ARROW:
            # Okay a bit complicated since arrows should be built from right to left
            # Notice than in no other case there might be a malformed arrow
            # therefore it happens at the top level
            # thus we can do it at the end (with the guarantee that the stack size will not be any lower than its current value from now on)
            # even more interesting part:
            # if the expression is well-formed then
            # we just need to put arrows between all elements of the stacks
            assert last_arrow + 1 == len(
                stack
            ), f"Invalid parsing: parsed:{stack} remaining:{text}"
            last_arrow += 1
        elif token == _TOK_OR:
            or_flag = 0

        # Manage or flags which consume things as they come
        if or_flag == 0:
            or_flag == 1
        elif or_flag == 1:
            or_flag = -1
            assert len(stack) >= 2
            stack.append(stack.pop() | stack.pop())

        # update text
        text = text[index:].strip()
    assert len(stack) >= 1
    while len(stack) > 1:
        last = stack.pop()
        stack.append(Arrow(stack.pop(), last))
    return stack.pop()
