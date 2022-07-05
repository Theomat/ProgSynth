from typing import Generator, List as TList, Optional, Tuple, Dict, Set, Iterable

from synth.syntax import Type, Arrow, List, PrimitiveType, PolymorphicType, FunctionType


SYMBOL_ANYTHING = "*"
SYMBOL_VAR_EXPR = "$"
SYMBOL_FORBIDDEN = "^"
SYMBOL_DUPLICATA = "@"
SYMBOL_SEPARATOR = ","

PREFIX_CAST = "cast#"
# ========================================================================================
# PARSING
# ========================================================================================
def parse_choices(expression: str) -> TList[str]:
    expression = expression.replace(" ", "")
    if "," in expression:
        return [s for s in expression.split(SYMBOL_SEPARATOR) if len(s) > 0]
    if len(expression) > 0:
        return [expression]
    return []


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


def parse_specification(spec: str) -> TList[str]:
    spec = spec.replace("\n", "").strip(")(")
    index = 0
    elements = []
    while index < len(spec):
        spec = spec[index:]
        word, index = __parse_next_word__(spec)
        elements.append(word)
    return elements


# ========================================================================================
# TYPE PRODUCERS/CONSUMERS
# ========================================================================================
def producers_of(syntax: Dict[str, Type], rtype: Type) -> Generator[str, None, None]:
    for prim, ptype in syntax.items():
        if isinstance(ptype, Arrow):
            if ptype.returns() == rtype:
                yield prim
        elif ptype == rtype:
            yield prim


def consumers_of(syntax: Dict[str, Type], atype: Type) -> Generator[str, None, None]:
    for prim, ptype in syntax.items():
        if isinstance(ptype, Arrow) and atype in ptype.arguments():
            yield prim


def producers_of_using(
    syntax: Dict[str, Type], rtype: Type, consuming: Set[Type]
) -> Set[str]:
    """
    Return the list of producers of <rtype> that can directly or indirectly consume any <consuming>
    """
    # Compute the list of all possible candidates from a type
    candidates = set()
    queue = [rtype]
    types_dones = set()
    while queue:
        atype = queue.pop()
        for prod in producers_of(syntax, atype):
            if prod not in candidates:
                candidates.add(prod)
                if isinstance(syntax[prod], Arrow):
                    for a in syntax[prod].arguments():
                        if a not in types_dones:
                            types_dones.add(a)
                            queue.append(a)

    # Compute all consumers
    all_consumers = set()
    for atype in consuming:
        all_consumers |= set(consumers_of(syntax, atype))
    # Now we can go down
    out = set(candidates).intersection(all_consumers)
    current = [p for p in out]
    types_dones = {x for x in consuming}
    while current:
        p = current.pop()
        ptype = syntax[p]
        if isinstance(ptype, Arrow):
            prtype = ptype.returns()
            if prtype not in types_dones:
                consumers = consumers_of(syntax, prtype)
                for consumer in consumers:
                    if consumer in candidates:
                        current.append(consumer)
                        out.add(consumer)
                types_dones.add(prtype)

    return out


def types_produced_directly_by(
    primitives: Iterable[str], syntax: Dict[str, Type]
) -> Set[Type]:
    out = set()
    for prim in primitives:
        ptype = syntax[prim]
        if isinstance(ptype, Arrow):
            out.add(ptype.returns())
        else:
            out.add(ptype)
    return out


def types_used_by(primitives: Iterable[str], syntax: Dict[str, Type]) -> Set[Type]:
    """
    Return the set of types that can be produced or consumed directly with the given primitives, then add all producers of those types recursively.

    """
    out = set()
    queue = []
    # Add all types from primitives
    for prim in primitives:
        ptype = syntax[prim]
        if isinstance(ptype, Arrow):
            for atype in [ptype.returns()] + ptype.arguments():
                if atype not in out:
                    out.add(atype)
                    queue.append(atype)
        else:
            if ptype not in out:
                out.add(ptype)
                queue.append(ptype)
    # Update list for all other types
    while queue:
        producers = producers_of(syntax, queue.pop())
        for prim in producers:
            ptype = syntax[prim]
            if isinstance(ptype, Arrow):
                for atype in [ptype.returns()] + ptype.arguments():
                    if atype not in out:
                        out.add(atype)
                        queue.append(atype)
            else:
                if ptype not in out:
                    out.add(ptype)
                    queue.append(ptype)
    return out


# ========================================================================================
# MISC
# ========================================================================================
def get_prefix(name: str) -> str:
    if name.startswith(PREFIX_CAST):
        return name
    return (
        name if SYMBOL_DUPLICATA not in name else name[: name.index(SYMBOL_DUPLICATA)]
    )


def map_type(type: Type, map: Dict[Type, Type]) -> Type:
    if type in map:
        return map[type]
    elif isinstance(type, List):
        return List(map_type(type.element_type, map))
    elif isinstance(type, Arrow):
        return FunctionType(
            *[map_type(arg, map) for arg in type.arguments()],
            map_type(type.returns(), map),
        )
    return type


def equivalent_primitives(syntax: Dict[str, Type], prefix: str) -> TList[str]:
    return [
        s
        for s in syntax.keys()
        if s.startswith(prefix)
        and (len(s) == len(prefix) or s[len(prefix)] == SYMBOL_DUPLICATA)
    ]


# ========================================================================================
# DUPLICATE PRIMITIVE/TYPE
# ========================================================================================


def __new_type_name__(name: str, syntax: Dict[str, Type]) -> str:
    all_types = set()
    for t in syntax.values():
        all_types |= {tt.type_name for tt in t.decompose_type()[0]}
    i = 0
    name = get_prefix(name)
    while f"{name}{SYMBOL_DUPLICATA}{i}" in all_types:
        i += 1

    return f"{name}{SYMBOL_DUPLICATA}{i}"


def duplicate_type(base: Type, syntax: Dict[str, Type]) -> Type:
    if isinstance(base, PrimitiveType):
        return PrimitiveType(__new_type_name__(base.type_name, syntax))
    elif isinstance(base, List):
        return List(duplicate_type(base.element_type, syntax))
    elif isinstance(base, PolymorphicType):
        # Not sure how relevnt this is
        return PolymorphicType(__new_type_name__(base.name, syntax))
    assert False


def __new_primitive_name__(primitive: str, syntax: Dict[str, Type]) -> str:
    i = 0
    primitive = get_prefix(primitive)
    while f"{primitive}{SYMBOL_DUPLICATA}{i}" in syntax:
        i += 1
    return f"{primitive}{SYMBOL_DUPLICATA}{i}"


def duplicate_primitive(primitive: str, syntax: Dict[str, Type]) -> str:
    new_name = __new_primitive_name__(primitive, syntax)
    syntax[new_name] = syntax[primitive]
    return new_name


# ========================================================================================
# CLEANING
# ========================================================================================


def __are_equivalent_types__(syntax: Dict[str, Type], t1: Type, t2: Type) -> bool:
    # Two types are equivalent iff
    #   for all primitives P producing t1 there is an equivalent primitive producing t2 and vice versa
    #   an equivalent primitive is a primitive that has the same type request but for the produced type and the same name_prefix up to @
    t2_producers = list(producers_of(syntax, t2))
    t1_producers = list(producers_of(syntax, t1))
    marked = [False for _ in range(len(t2_producers))]
    # t1 in t2
    for p1 in t1_producers:
        found_match = False
        for i, p2 in enumerate(t2_producers):
            if get_prefix(p1) != get_prefix(p2):
                continue
            if isinstance(syntax[p1], Arrow) and isinstance(syntax[p2], Arrow):
                found_match = syntax[p1].arguments() == syntax[p2].arguments()
                if found_match:
                    marked[i] = True
                    break
            if not isinstance(syntax[p1], Arrow) and not isinstance(syntax[p2], Arrow):
                found_match = True
                marked[i] = True
                break
        if not found_match:
            return False
    # t2 in t1
    for already_found, p1 in zip(marked, t2_producers):
        if already_found:
            continue
        found_match = False
        for p2 in t1_producers:
            if get_prefix(p1) != get_prefix(p2):
                continue
            if isinstance(syntax[p1], Arrow) and isinstance(syntax[p2], Arrow):
                found_match = syntax[p1].arguments() == syntax[p2].arguments()
                if found_match:
                    break
            if not isinstance(syntax[p1], Arrow) and not isinstance(syntax[p2], Arrow):
                found_match = True
                break
        if not found_match:
            return False
    return True


def __replace_type__(syntax: Dict[str, Type], old_t: Type, new_t: Type):
    tmap = {old_t: new_t}
    for P, ptype in syntax.items():
        syntax[P] = map_type(ptype, tmap)


def __merge_for__(syntax: Dict[str, Type], primitive: str) -> bool:
    candidates = equivalent_primitives(syntax, primitive)
    if len(candidates) <= 1:
        return False

    merged = False
    # Handle terminal
    if not isinstance(syntax[candidates[0]], Arrow):
        # Delete those with no consumers
        for P in candidates:
            if not any(consumers_of(syntax, syntax[P])):
                merged = True
                del syntax[P]
    # Merge those with same types
    for i, P1 in enumerate(candidates):
        if P1 not in syntax:
            continue
        for P2 in candidates[i + 1 :]:
            if P2 not in syntax:
                continue
            if syntax[P1] == syntax[P2]:
                del syntax[P2]
                merged = True

    return merged


def clean(syntax: Dict[str, Type], type_request: Optional[Arrow] = None) -> None:
    """
    Try merging duplicates that were created.
    """
    # Delete all primitives using a non-interesting type
    all_primitives = set(get_prefix(p) for p in syntax.keys())
    interesting_types = types_used_by(all_primitives, syntax)
    var_types = set()
    if type_request and isinstance(type_request, Arrow):
        var_types = set(type_request.arguments())
    interesting_types |= var_types
    for P in list(syntax.keys()):
        ptype = syntax[P]
        if isinstance(ptype, Arrow) and (
            any(tt not in interesting_types for tt in ptype.arguments())
            or ptype.returns() not in interesting_types
        ):
            del syntax[P]
        elif not isinstance(ptype, Arrow) and ptype not in interesting_types:
            del syntax[P]
    # Gather equivalent(by name up to @) in groups
    type_classes = {}
    for t in interesting_types:
        prefix = get_prefix(str(t))
        if prefix in type_classes:
            continue
        type_classes[prefix] = [
            tt
            for tt in interesting_types
            if get_prefix(str(tt)) == prefix and tt not in var_types
        ]

    merged_types = True
    while merged_types:
        merged_types = False

        for prefix, tclass in list(type_classes.items()):
            next_gen: TList[Type] = tclass[:]
            # Try to merge two types in equivalence class
            for i, t1 in enumerate(tclass):
                if t1 not in next_gen:
                    continue
                for t2 in tclass[i + 1 :]:
                    if t2 not in next_gen:
                        continue
                    if __are_equivalent_types__(syntax, t1, t2):
                        __replace_type__(syntax, t2, t1)
                        next_gen.remove(t2)
                        for p in all_primitives:
                            __merge_for__(syntax, p)
                        merged_types = True
            type_classes[prefix] = next_gen
