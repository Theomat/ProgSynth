from collections import defaultdict
from typing import Any, Optional, Dict, Iterable, Tuple, List as TList
from utils import (
    PREFIX_CAST,
    SYMBOL_VAR_EXPR,
    equivalent_primitives,
    map_type,
    producers_of,
    duplicate_primitive,
    duplicate_type,
    get_prefix,
    parse_choices,
    SYMBOL_SEPARATOR,
    SYMBOL_FORBIDDEN,
    SYMBOL_ANYTHING,
    clean,
    parse_specification,
    producers_of_using,
    types_produced_directly_by,
)

from synth.syntax import Type, Arrow


def __add_variable_constraint__(
    content: str,
    parent: str,
    argno: int,
    arg_type: Type,
    syntax: Dict[str, Type],
    nconstraints: Dict[str, int],
    type_request: Optional[Arrow],
    level: int = 0,
) -> Arrow:
    assert type_request, "A type request is needed for variable constraints!"
    content = content.strip(f"{SYMBOL_VAR_EXPR}()")
    variables = set(map(int, parse_choices(content.replace("var", ""))))
    var_types = set(type_request.arguments()[i] for i in variables)
    varno2type = {no: type_request.arguments()[no] for no in variables}
    # print("\t" * level,  "Variables:", variables, "types:", var_types)
    # Always assume there are other variables of the same types

    to_duplicate = producers_of_using(syntax, arg_type, var_types)
    # add constantsof same types
    to_duplicate |= set(
        [
            p
            for p, ptype in syntax.items()
            if not isinstance(ptype, Arrow) and ptype in var_types
        ]
    )
    types_to_duplicate = types_produced_directly_by(to_duplicate, syntax)
    # print("\t" * level,  "To duplicate:", to_duplicate)
    # print("\t" * level,  "Types to duplicate:", types_to_duplicate)

    # Compute the mapping of types
    types_map = {}
    # variables first
    for var_type in var_types:
        concerned = {i for i in variables if varno2type[i] == var_type}
        if any(nconstraints[f"var{i}"] > 0 for i in concerned):
            already_defined = [i for i in concerned if nconstraints[f"var{i}"] > 0]
            assert len(already_defined) == 1
            types_map[var_type] = var_type
        else:
            types_map[var_type] = duplicate_type(var_type, syntax)
    # rest
    for type in types_to_duplicate.difference(var_types):
        types_map[type] = duplicate_type(type, syntax)

    # Duplicate primitives
    for primitive in to_duplicate:
        new_primitive = duplicate_primitive(primitive, syntax)
        syntax[new_primitive] = map_type(syntax[new_primitive], types_map)

    # Add casts
    for var_type in var_types:
        syntax[f"{PREFIX_CAST}{types_map[var_type]}->{var_type}"] = FunctionType(
            types_map[var_type], var_type
        )

    # Fix parent type
    args = syntax[parent].arguments()
    args[argno] = map_type(args[argno], types_map)
    syntax[parent] = FunctionType(*args, syntax[parent].returns())

    # Fix type request
    args = type_request.arguments()
    for i in variables:
        args[i] = map_type(args[i], types_map)
        # Add constraints
        nconstraints[f"var{i}"] += 1
    return FunctionType(*args, type_request.returns())


def __add_primitive_constraint__(
    content: str,
    parent: str,
    argno: int,
    syntax: Dict[str, Type],
    level: int = 0,
    target_type: Optional[Type] = None,
) -> str:
    prim = content.strip("()")
    for primitive in equivalent_primitives(syntax, prim):
        ptype = syntax[primitive]
        rtype = ptype.returns() if isinstance(ptype, Arrow) else ptype
        new_type_needed = any(
            get_prefix(p) != get_prefix(prim) for p in producers_of(syntax, rtype)
        )
        # If there are other ways to produce the same thing
        if new_type_needed:
            new_primitive = duplicate_primitive(primitive, syntax)
            new_return_type = (
                duplicate_type(rtype, syntax) if target_type is None else target_type
            )
            if isinstance(ptype, Arrow):
                syntax[new_primitive] = FunctionType(
                    *ptype.arguments(), new_return_type
                )
            else:
                syntax[new_primitive] = new_return_type

            if primitive == prim:
                prim = new_primitive
            # print("\t" * level, "Added:", new_primitive, ":", syntax[primitive])
            # Update parent signature
            parent_type = syntax[parent]
            assert isinstance(parent_type, Arrow)
            old_types = parent_type.arguments() + [parent_type.returns()]
            old_types[argno] = new_return_type
            syntax[parent] = FunctionType(*old_types)
    return prim


def __add_primitives_constraint__(
    content: str,
    parent: str,
    argno: int,
    syntax: Dict[str, Type],
    level: int = 0,
) -> None:
    primitives = parse_choices(content)
    if len(primitives) <= 0:
        return
    # print("\t" * level, "\tcontent:", content)
    # print("\t" * level, "\tprimitives:", primitives)
    # 1) Simply do it for all primitives in the list
    new_primitives = [
        __add_primitive_constraint__(p, parent, argno, syntax, level)
        for p in primitives
    ]
    # 2) Make them coherent
    ttype = syntax[new_primitives[0]]
    if isinstance(ttype, Arrow):
        ttype = ttype.returns()
    for new_primitive in new_primitives:
        ntype = syntax[new_primitive]
        syntax[new_primitive] = (
            FunctionType(*ntype.arguments(), ttype)
            if isinstance(ntype, Arrow)
            else ttype
        )
    # Update parent signature
    parent_type = syntax[parent]
    assert isinstance(parent_type, Arrow)
    old_types = parent_type.arguments() + [parent_type.returns()]
    old_types[argno] = ttype
    syntax[parent] = FunctionType(*old_types)

    # Now small thing to take into account
    # if parent is the same as one of our primitive we need to fix the children
    for p in new_primitives:
        if get_prefix(p) == get_prefix(parent):
            ptype = syntax[p]
            assert isinstance(ptype, Arrow)
            old_types = ptype.arguments() + [ptype.returns()]
            old_types[argno] = ttype
            syntax[p] = FunctionType(*old_types)


def __add_forbidden_constraint__(
    content: str,
    parent: str,
    argno: int,
    syntax: Dict[str, Type],
    *args: Any,
    level: int = 0,
    **kwargs: Any,
):
    # print("\t" * level, "\tcontent:", content)
    primitives = parse_choices(content[1:])
    all_forbidden = set()
    for p in primitives:
        all_forbidden |= set(equivalent_primitives(syntax, p))
    all_producers = producers_of(syntax, syntax[parent].arguments()[argno])
    remaining = all_producers - all_forbidden
    # print("\t" * level, "\tallowed:", remaining)

    __add_primitives_constraint__(
        SYMBOL_SEPARATOR.join(remaining), parent, argno, syntax, *args, **kwargs
    )


def __process__(
    constraint: TList[str],
    syntax: Dict[str, Type],
    nconstraints: Dict[str, int],
    type_request: Arrow,
    level: int = 0,
) -> Tuple[TList[str], Arrow]:
    # If one element then there is nothing to do.
    if len(constraint) == 1:
        return constraint, type_request
    function = equivalent_primitives(syntax, constraint.pop(0))
    args = []
    # We need to process all arguments first
    for arg in constraint:
        new_el, type_request = __process__(
            parse_specification(arg), syntax, nconstraints, type_request, level + 1
        )
        args.append(new_el)
    # If there are only stars there's nothing to do at our level
    if all(len(arg) == 1 and arg[0] == "*" for arg in args):
        return function, type_request

    # print("\t" * level, "processing:", constraint)
    for parent in function:
        fun_tr = syntax[get_prefix(parent)]
        assert isinstance(fun_tr, Arrow)
        for argno, (eq_args, arg_type) in enumerate(zip(args, fun_tr.arguments())):
            if len(eq_args) > 1:
                __add_primitives_constraint__(
                    content, parent, argno, syntax, nconstraints, level
                )
            else:
                content: str = eq_args[0]
                if content == SYMBOL_ANYTHING:
                    continue
                elif content[0] == SYMBOL_VAR_EXPR:
                    type_request = __add_variable_constraint__(
                        content,
                        parent,
                        argno,
                        arg_type,
                        syntax,
                        nconstraints,
                        type_request,
                        level,
                    )
                elif content[0] == SYMBOL_FORBIDDEN:
                    __add_forbidden_constraint__(content, parent, argno, syntax, level)
                else:
                    __add_primitives_constraint__(content, parent, argno, syntax, level)

    return function, type_request


def produce_new_syntax_for_constraints(
    syntax: Dict[str, Type],
    constraints: Iterable[str],
    type_request: Optional[Arrow] = None,
) -> Tuple[Dict[str, Type], Optional[Arrow]]:
    """
    Add type constraints on the specified syntax in order to enforce the given constraints.

    If no constraint depends on variables the type request is ignored.
    """
    new_syntax = {k: v for k, v in syntax.items()}
    parsed_constraints = [parse_specification(constraint) for constraint in constraints]
    for constraint in parsed_constraints:
        _, type_request = __process__(
            constraint, new_syntax, defaultdict(int), type_request
        )
    return new_syntax, type_request


if __name__ == "__main__":
    from synth.syntax import DSL, ConcreteCFG, INT, FunctionType, ConcretePCFG
    from examples.pbe.towers.towers_base import syntax, BLOCK

    type_request = FunctionType(INT, INT, BLOCK)

    patterns = [
        "and ^and *",
        "or ^or,and ^and",
        "+ ^+ *",
        # "elif if elif,if,EMPTY",
        "ifY * 1x3,3x1",
        "ifX $(var0) ifY,elifY",
        "elifY ifY EMPTY,elifY",
        "elifX ifX EMPTY,elifX",
        # "elif ^EMPTY elif,if,EMPTY",
        # "elif ^elif *"
        # "elif ^EMPTY,elif elif,if,EMPTY",
    ]
    max_depth = 5
    original_size = ConcreteCFG.from_dsl(DSL(syntax), type_request, max_depth).size()

    # test for patterns
    new_syntax = syntax
    new_syntax, type_request = produce_new_syntax_for_constraints(
        new_syntax, patterns, type_request
    )
    # Print
    print(f"[BEF CLEAN][PATTERNS] New syntax ({len(new_syntax)} primitives):")
    for prim, type in new_syntax.items():
        print("\t", prim, ":", type)
    new_size = ConcreteCFG.from_dsl(DSL(new_syntax), type_request, max_depth).size()
    pc = (original_size - new_size) / original_size
    print("Removed", original_size - new_size, f"({pc:%}) programs at depth", max_depth)
    print("New TR:", type_request)

    clean(new_syntax, type_request)
    print(f"[AFT CLEAN][PATTERNS] New syntax ({len(new_syntax)} primitives):")
    for prim, type in new_syntax.items():
        print("\t", prim, ":", type)

    new_size = ConcreteCFG.from_dsl(DSL(new_syntax), type_request, max_depth).size()
    pc = (original_size - new_size) / original_size
    print("Removed", original_size - new_size, f"({pc:%}) programs at depth", max_depth)

    pcfg = ConcretePCFG.uniform(
        ConcreteCFG.from_dsl(DSL(new_syntax), type_request, max_depth)
    )
    pcfg.init_sampling(2)
    for i in range(30):
        print(pcfg.sample_program())
