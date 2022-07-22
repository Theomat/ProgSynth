from collections import defaultdict
from typing import Optional, Dict, Set, Tuple, List as TList

from synth.syntax import Type, Arrow, FunctionType
from synth.pruning.type_constraints.utils import (
    SYMBOL_VAR_EXPR,
    Syntax,
    SYMBOL_SEPARATOR,
    SYMBOL_FORBIDDEN,
    SYMBOL_ANYTHING,
    clean,
    parse_specification,
)
from synth.pruning.type_constraints.pattern_constraints import (
    __process__ as __pat_process__,
    __add_forbidden_constraint__,
    __add_primitives_constraint__,
    __add_variable_constraint__,
)


def __process__(
    constraint: TList[str],
    syntax: Syntax,
    nconstraints: Dict[str, int],
    type_request: Arrow,
) -> Tuple[TList[str], Arrow]:
    # If one element then there is nothing to do.
    if len(constraint) == 1:
        return constraint, type_request
    function = sorted(syntax.equivalent_primitives(constraint.pop(0)))
    args = []

    old_rtype = type_request.returns()
    new_rtype = syntax.duplicate_type(type_request.returns())
    type_request = FunctionType(*type_request.arguments(), new_rtype)  # type: ignore

    new_function = []
    for parent in function:
        fun_tr = syntax[parent]
        if fun_tr.returns() != old_rtype:
            continue
        parent = syntax.duplicate_primitive(
            parent, FunctionType(*fun_tr.arguments(), new_rtype)
        )
        fun_tr = syntax[parent]
        new_function.append(parent)
    function = new_function

    # We need to process all arguments first
    for arg in constraint:
        new_el, tr = __pat_process__(
            parse_specification(arg), syntax, nconstraints, function, type_request, 1
        )  #
        assert tr is not None
        type_request = tr
        args.append(new_el)
    # If there are only stars there's nothing to do at our level
    if all(len(arg) == 1 and arg[0] == SYMBOL_ANYTHING for arg in args):
        return function, type_request
    # print("\t" * level, "processing:", constraint)
    for parent in function:
        fun_tr = syntax[parent]
        assert isinstance(fun_tr, Arrow)
        for argno, (eq_args, arg_type) in enumerate(zip(args, fun_tr.arguments())):
            if len(eq_args) > 1:
                __add_primitives_constraint__(
                    SYMBOL_SEPARATOR.join(eq_args), parent, argno, syntax, 0
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
                        0,
                    )
                elif content[0] == SYMBOL_FORBIDDEN:
                    __add_forbidden_constraint__(content, parent, argno, syntax, 0)
                else:
                    __add_primitives_constraint__(content, parent, argno, syntax, 0)

    return function, type_request


def produce_new_syntax_for_sketch(
    syntax: Dict[str, Type],
    sketch: str,
    type_request: Arrow,
    forbidden: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[Dict[str, Type], Arrow]:
    """
    Add type constraints on the specified syntax in order to enforce the given sketch.
    """
    new_syntax = Syntax({k: v for k, v in syntax.items()}, forbidden)
    sketch_spec = parse_specification(sketch)
    _, type_request = __process__(
        sketch_spec, new_syntax, defaultdict(int), type_request
    )
    print(type_request)
    clean(new_syntax, type_request)
    return new_syntax.syntax, type_request


if __name__ == "__main__":
    from synth.syntax import DSL, CFG, INT, FunctionType, ProbDetGrammar, List
    from synth.pruning.type_constraints.utils import export_syntax_to_python

    from examples.pbe.deepcoder.deepcoder import dsl, __primitive_types  # type: ignore

    type_request = FunctionType(List(INT), List(INT))

    max_depth = 6
    # original_size = CFG.depth_constraint(DSL(syntax), type_request, max_depth).size()
    original_size = CFG.depth_constraint(dsl, type_request, max_depth).size()

    new_syntax, type_request = produce_new_syntax_for_sketch(
        # __primitive_types,
        # "(MAP[*2] (MAP[+1] _))",
        # type_request
        __primitive_types,
        "(MAP[*2] (MAP[+1] (MAP[*2] _)))",
        type_request,  # type: ignore
    )

    # Print
    print(f"[PATTERNS] New syntax with {len(dsl.list_primitives)} primitives")
    for P in DSL(new_syntax).list_primitives:
        prim, type = P.primitive, P.type
        print("\t", prim, ":", type)
    new_size = CFG.depth_constraint(
        # dsl,
        # type_request,
        # max_depth
        DSL(new_syntax),
        type_request,
        max_depth,
    ).size()
    pc = (original_size - new_size) / original_size
    print(
        f"Removed {original_size - new_size:.2E} ({pc:%}) programs at depth", max_depth
    )
    print(f"New size {new_size:.2E} programs at depth", max_depth)
    print("New TR:", type_request)

    pcfg = ProbDetGrammar.uniform(
        CFG.depth_constraint(DSL(new_syntax), type_request, max_depth)
    )
    pcfg.init_sampling(20)
    for i in range(30):
        print(pcfg.sample_program())

    # with open("deepcoder2.py", "w") as fd:
    # fd.write(export_syntax_to_python(new_syntax))
