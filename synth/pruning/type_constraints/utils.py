from collections import defaultdict
from typing import Generator, List as TList, Optional, Tuple, Dict, Set, Iterable, Union


from synth.syntax import Type, Arrow, List, PrimitiveType, PolymorphicType, FunctionType


SYMBOL_ANYTHING = "_"
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


class Syntax:
    def __init__(
        self,
        type_constraints: Dict[str, Type],
        forbidden: Optional[Dict[Tuple[str, int], Set[str]]] = None,
    ) -> None:
        self.syntax = type_constraints
        self.forbidden_patterns = forbidden or {}

        self._new_types_index: Dict[str, int] = defaultdict(int)
        for ttype in __all_types__(self.syntax):
            name = str(ttype)
            if "@" in name and not name.startswith(PREFIX_CAST):
                id = int(name[name.index("@") + 1 :])
                self._new_types_index[get_prefix(name)] = id + 1

        # Init producers by type
        self.producers_by_type: Dict[Type, Set[str]] = defaultdict(set)
        for prim, ptype in self.syntax.items():
            if isinstance(ptype, Arrow):
                self.producers_by_type[ptype.returns()].add(prim)
            else:
                self.producers_by_type[ptype].add(prim)

        # Init equivalent primitives
        self.equivalents: Dict[str, Set[str]] = defaultdict(set)
        for prim in self.syntax:
            self.equivalents[get_prefix(prim)].add(prim)

    def __getitem__(self, item: str) -> Type:
        return self.syntax[item]

    def __contains__(self, item: str) -> bool:
        return item in self.syntax

    def __len__(self) -> int:
        return len(self.syntax)

    def __delitem__(self, item: str) -> None:
        ptype = self.syntax[item]
        rtype = ptype
        if isinstance(ptype, Arrow):
            rtype = ptype.returns()
        self.producers_by_type[rtype].remove(item)
        self.equivalents[get_prefix(item)].remove(item)
        del self.syntax[item]

    def __setitem__(self, item: str, new_t: Type) -> None:
        ptype = self.syntax[item]
        rtype = ptype.returns() if isinstance(ptype, Arrow) else ptype
        rtype_now = new_t.returns() if isinstance(new_t, Arrow) else new_t
        if rtype != rtype_now:
            self.producers_by_type[rtype].remove(item)
            self.producers_by_type[rtype_now].add(item)

        self.syntax[item] = new_t

    def producers_of(self, rtype: Type) -> Set[str]:
        return self.producers_by_type[rtype]

    def consumers_of(self, atype: Type) -> Generator[str, None, None]:
        for prim, ptype in self.syntax.items():
            if isinstance(ptype, Arrow) and atype in ptype.arguments():
                yield prim

    def equivalent_primitives(self, name: str) -> Set[str]:
        return self.equivalents[get_prefix(name)]

    def replace_type(self, old_t: Type, new_t: Type) -> None:
        tmap = {old_t: new_t}
        for P, ptype in self.syntax.items():
            self.syntax[P] = map_type(ptype, tmap)
            if P in self.producers_by_type[old_t]:
                self.producers_by_type[old_t].remove(P)
                self.producers_by_type[new_t].add(P)

    def duplicate_primitive(self, primitive: str, ntype: Type) -> str:
        new_name = __new_primitive_name__(primitive, self)
        self.syntax[new_name] = ntype
        self.equivalents[get_prefix(new_name)].add(new_name)
        rtype = ntype
        if isinstance(ntype, Arrow):
            rtype = ntype.returns()
        self.producers_by_type[rtype].add(new_name)
        return new_name

    def __new_type_name__(self, name: str) -> str:
        prefix = get_prefix(name)
        id = self._new_types_index[prefix]
        self._new_types_index[prefix] += 1
        return f"{prefix}{SYMBOL_DUPLICATA}{id}"

    def duplicate_type(self, base: Type) -> Type:
        out: Optional[Type] = None
        if isinstance(base, PrimitiveType):
            out = PrimitiveType(self.__new_type_name__(base.type_name))
        elif isinstance(base, List):
            out = List(self.duplicate_type(base.element_type))
        elif isinstance(base, PolymorphicType):
            # Not sure how relevant this is
            out = PolymorphicType(self.__new_type_name__(base.name))
        assert out is not None, f"Could not duplicate type:{base}"
        return out

    def add_cast(self, from_type: Type, to: Type) -> None:
        name = f"{PREFIX_CAST}{from_type}->{to}"
        self.syntax[name] = Arrow(from_type, to)
        self.producers_by_type[to].add(name)

    def filter_out_forbidden(
        self, parent: str, argno: int, all_forbidden: Iterable[str]
    ) -> TList[str]:
        forbidden = self.forbidden_patterns.get((parent, argno), set())
        return [P for P in all_forbidden if get_prefix(P) not in forbidden]


def producers_of_using(syntax: Syntax, rtype: Type, consuming: Set[Type]) -> Set[str]:
    """
    Return the list of producers of <rtype> that can directly or indirectly consume any <consuming>
    """
    # Compute the list of all possible candidates from a type
    candidates = set()
    queue = [rtype]
    types_dones = set()
    while queue:
        atype = queue.pop()
        for prod in syntax.producers_of(atype):
            if prod not in candidates:
                candidates.add(prod)
                ptype = syntax[prod]
                if isinstance(ptype, Arrow):
                    for a in ptype.arguments():
                        if a not in types_dones:
                            types_dones.add(a)
                            queue.append(a)

    # Compute all consumers
    all_consumers: Set[str] = set()
    for atype in consuming:
        all_consumers |= set(syntax.consumers_of(atype))
    # Now we can go down
    out: Set[str] = candidates.intersection(all_consumers)
    current: TList[str] = list(out)
    types_dones = {x for x in consuming}
    while current:
        p = current.pop()
        ptype = syntax[p]
        if isinstance(ptype, Arrow):
            prtype = ptype.returns()
            if prtype not in types_dones:
                consumers = syntax.consumers_of(prtype)
                for consumer in consumers:
                    if consumer in candidates:
                        current.append(consumer)
                        out.add(consumer)
                types_dones.add(prtype)

    return out


def types_produced_directly_by(primitives: Iterable[str], syntax: Syntax) -> Set[Type]:
    out = set()
    for prim in primitives:
        ptype = syntax[prim]
        if isinstance(ptype, Arrow):
            out.add(ptype.returns())
        else:
            out.add(ptype)
    return out


def types_used_by(primitives: Iterable[str], syntax: Syntax) -> Set[Type]:
    """
    Return the set of types that can be produced or consumed directly with the given primitives, then add all producers of those types recursively.

    """
    out = set()
    queue = []
    # Add all types from primitives
    for prim in primitives:
        if prim not in syntax:
            continue
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
        producers = syntax.producers_of(queue.pop())
        for prim in producers:
            ptype = syntax[prim]
            if isinstance(ptype, Arrow):
                for atype in ptype.arguments():
                    if atype not in out:
                        out.add(atype)
                        queue.append(atype)
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


def map_type(old_type: Type, map: Dict[Type, Type]) -> Type:
    if old_type in map:
        return map[old_type]
    elif isinstance(old_type, List):
        return List(map_type(old_type.element_type, map))
    elif isinstance(old_type, Arrow):
        return FunctionType(
            *[map_type(arg, map) for arg in old_type.arguments()],
            map_type(old_type.returns(), map),
        )
    return old_type


# ========================================================================================
# DUPLICATE PRIMITIVE/TYPE
# ========================================================================================


def __all_types__(syntax: Dict[str, Type]) -> Set[PrimitiveType]:
    all_types: Set[PrimitiveType] = set()
    for t in syntax.values():
        for tt in t.decompose_type()[0]:
            all_types.add(tt)
    return all_types


def __new_primitive_name__(primitive: str, syntax: Syntax) -> str:
    i = 0
    primitive = get_prefix(primitive)
    while f"{primitive}{SYMBOL_DUPLICATA}{i}" in syntax:
        i += 1
    return f"{primitive}{SYMBOL_DUPLICATA}{i}"


# ========================================================================================
# CLEANING
# ========================================================================================


def __are_equivalent_types__(syntax: Syntax, t1: Type, t2: Type) -> bool:
    # Two types are equivalent iff
    #   for all primitives P producing t1 there is an equivalent primitive producing t2 and vice versa
    #   an equivalent primitive is a primitive that has the same type request but for the produced type and the same name_prefix up to @
    t2_producers = syntax.producers_of(t2)
    t1_producers = syntax.producers_of(t1)
    marked = [False for _ in range(len(t2_producers))]
    # t1 in t2
    for p1 in t1_producers:
        found_match = False
        for i, p2 in enumerate(t2_producers):
            if get_prefix(p1) != get_prefix(p2):
                continue
            tp1 = syntax[p1]
            tp2 = syntax[p2]
            if isinstance(tp1, Arrow) and isinstance(tp2, Arrow):
                found_match = tp1.arguments() == tp2.arguments()
                if found_match:
                    marked[i] = True
                    break
            if not isinstance(tp1, Arrow) and not isinstance(tp2, Arrow):
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
            tp1 = syntax[p1]
            tp2 = syntax[p2]
            if isinstance(tp1, Arrow) and isinstance(tp2, Arrow):
                found_match = tp1.arguments() == tp2.arguments()
                if found_match:
                    break
            if not isinstance(tp1, Arrow) and not isinstance(tp2, Arrow):
                found_match = True
                break
        if not found_match:
            return False
    return True


def __merge_for__(syntax: Syntax, primitive: str) -> bool:
    candidates = sorted(syntax.equivalent_primitives(primitive))
    if len(candidates) <= 1:
        return False

    merged = False
    # Handle terminal
    if not isinstance(syntax[candidates[0]], Arrow):
        # Delete those with no consumers
        for P in candidates:
            if not any(syntax.consumers_of(syntax[P])):
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


def clean(
    syntax: Union[Syntax, Dict[str, Type]], type_request: Optional[Arrow] = None
) -> None:
    """
    Try merging duplicates that were created.
    """
    if not isinstance(syntax, Syntax):
        syntax = Syntax(syntax)
    assert isinstance(syntax, Syntax)
    # Delete all primitives using a non-interesting type
    all_primitives = set(syntax.equivalents.keys())
    interesting_types = types_used_by(all_primitives, syntax)
    var_types = set()
    if type_request and isinstance(type_request, Arrow):
        var_types = set(type_request.arguments())
        var_types.add(type_request.returns())
    interesting_types |= var_types
    deletable_candidates = set(syntax.syntax.keys())
    ntypes = len(interesting_types)
    old_n = 0
    while ntypes > old_n:
        old_n = ntypes
        new_deletable = set()
        for P in deletable_candidates:
            ptype = syntax[P]
            # P cannot be used
            if any(tt not in interesting_types for tt in ptype.arguments()):
                new_deletable.add(P)
            else:
                for tt in ptype.arguments():
                    interesting_types.add(tt)
                interesting_types.add(ptype.returns())
        deletable_candidates = new_deletable
        ntypes = len(interesting_types)
    # Now we are sure that these primitives can be deleted
    for P in deletable_candidates:
        del syntax[P]

    # Do one merge for each primitive (only useful if no type merge occurs)
    for p in all_primitives:
        __merge_for__(syntax, p)

    # Gather equivalent type (by name up to @) in groups
    type_classes = {}
    for t in interesting_types:
        prefix = get_prefix(str(t))
        if prefix in type_classes:
            continue
        type_classes[prefix] = sorted(
            [
                tt
                for tt in interesting_types
                if get_prefix(str(tt)) == prefix and tt not in var_types
            ],
            key=str,
        )
    # Try merging types
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
                        syntax.replace_type(t2, t1)
                        next_gen.remove(t2)
                        for p in all_primitives:
                            __merge_for__(syntax, p)
                        merged_types = True
            type_classes[prefix] = next_gen


# ========================================================================================
# EXPORTING
# ========================================================================================
def __export_type__(ptype: Type) -> str:
    if isinstance(ptype, Arrow):
        return f"Arrow({__export_type__(ptype.type_in)}, {__export_type__(ptype.type_out)})"
    elif isinstance(ptype, PrimitiveType):
        return ptype.type_name.replace("@", "_")
    elif isinstance(ptype, List):
        return f"List({__export_type__(ptype.element_type)})"
    elif isinstance(ptype, PolymorphicType):
        return f"PolymorphicType({ptype.name})"
    assert False


def export_syntax_to_python(syntax: Dict[str, Type], varname: str = "syntax") -> str:
    nsyntax = Syntax(syntax)
    types_declaration = ""
    types = types_used_by(nsyntax.syntax.keys(), nsyntax)
    for ntype in types:
        while isinstance(ntype, List):
            ntype = ntype.element_type
        if isinstance(ntype, PrimitiveType) and "@" in ntype.type_name:
            types_declaration += (
                ntype.type_name.replace("@", "_")
                + ' = PrimitiveType("'
                + ntype.type_name
                + '")'
                + "\n"
            )

    out = f"{varname} = " + "{\n"
    for prim, ptype in syntax.items():
        out += '\t"' + prim + '": ' + __export_type__(ptype) + ",\n"
    out += "\n}"
    return types_declaration + out
