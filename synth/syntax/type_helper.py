"""
An helper file that contains useful methods to make type creation and manipulation very easy.

"""
from synth.syntax.type_system import FixedPolymorphicType, Generic, PrimitiveType
from synth.syntax.type_system import (
    Type,
    UnknownType,
    Arrow,
    EmptyList,
    INT,
    BOOL,
    STRING,
    UNIT,
    List,
    PolymorphicType,
)

from typing import Any, Dict, Tuple, Union, overload, List as TList


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
_TOK_INFIX = 2
_TOK_POLYMORPHIC = 3
_TOK_OR = 4


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


__SPECIAL_TOKENS = " ()"


def __next_token__(text: str) -> Tuple[str, int, int]:
    if text.startswith("(") or text.startswith("["):
        i = __matching__(text)
        return text[1:i], _TOK_BRACKETS if text[0] == "[" else _TOK_PARENTHESIS, i + 1
    elif text.startswith("|"):
        return "", _TOK_OR, 1
    elif not text[0].isalpha() and not text[0] == "'":
        i = 1
        while i < len(text) and not (text[i].isalpha() or text[i] in __SPECIAL_TOKENS):
            i += 1
        return text[:i], _TOK_INFIX, i
    i = 1
    while i < len(text) and text[i].isalpha():
        i += 1
    is_poly = len(text) > 0 and text[0] == "'"
    return (
        text[is_poly:i],
        _TOK_POLYMORPHIC if is_poly else _TOK_NONE,
        i,
    )


@overload
def auto_type(el: str) -> Type:
    """
    Automatically build the type from its string representation.

    Parameters:
    -----------
    - el: the type's string representation
    """
    pass


@overload
def auto_type(el: Dict[str, str]) -> Dict[str, Type]:
    """
    Automatically build the type from its string representation for all values of the dictionnary.

    Parameters:
    -----------
    - el: a dictionnary containing as values types string representations
    """
    pass


def auto_type(el: Union[Dict[str, str], str]) -> Union[Dict[str, Type], Type]:
    # Dictionnary part
    if isinstance(el, dict):
        return {k: auto_type(v) for k, v in el.items()}
    # String part
    stack = []
    text = el
    last_infix = 0
    infix_stack = []
    or_flag = -1
    index = 1
    while len(text) > 0:
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
        elif token == _TOK_POLYMORPHIC:
            stack.append(PolymorphicType(w))
        elif token == _TOK_NONE:
            if len(w) > 0:
                if last_infix < len(stack) and or_flag < 0:
                    stack.append(Generic(w, stack.pop()))
                else:
                    stack.append(PrimitiveType(w))
        elif token == _TOK_INFIX:
            # old comment about arrows but the same for infix operators
            # Okay a bit complicated since arrows should be built from right to left
            # Notice than in no other case there might be a malformed arrow
            # therefore it happens at the top level
            # thus we can do it at the end (with the guarantee that the stack size will not be any lower than its current value from now on)
            # even more interesting part:
            # if the expression is well-formed then
            # we just need to put arrows between all elements of the stacks
            last_infix += 1
            infix_stack.append(w)
        elif token == _TOK_OR:
            or_flag = 0

        # Manage or flags which consume things as they come
        if or_flag == 0:
            or_flag = 1
        elif or_flag == 1:
            or_flag = -1
            assert len(stack) >= 2
            last = stack.pop()
            stack.append(stack.pop() | last)

        # update text
        text = text[index:].strip()
    assert len(stack) >= 1
    while len(stack) > 1:
        last = stack.pop()
        w = infix_stack.pop()
        if w == "->":
            stack.append(Arrow(stack.pop(), last))
        else:
            stack.append(Generic(w, stack.pop(), last, infix=True))
    return stack.pop()
