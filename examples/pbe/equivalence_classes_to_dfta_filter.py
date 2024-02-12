from typing import List, Dict, Optional, Set, Tuple
import itertools

import tqdm
from colorama import Fore as F

from synth.syntax import (
    Program,
    Constant,
    Primitive,
    Variable,
    Function,
    DSL,
    Type,
    DFTA,
    UnknownType,
)
from synth.filter import DFTAFilter
from synth.syntax.grammars.grammar import DerivableProgram

uk = UnknownType()


def dsl_2_dfta(dsl: DSL) -> DFTA[Tuple[Type, DerivableProgram], DerivableProgram]:
    # TODO: we are missing curried programs
    primitives = dsl.list_primitives
    primitive2state: Dict[Primitive, Tuple[Type, Primitive]] = {}
    rules: Dict[
        Tuple[str, Tuple[Tuple[Type, DerivableProgram], ...]],
        Tuple[Type, DerivableProgram],
    ] = {}
    for primitive in primitives:
        primitive2state[primitive] = (primitive.type.returns(), primitive)
    for primitive in primitives:
        args_possibles = []
        for arg_type in primitive.type.arguments():
            args_possibles.append(
                [
                    primitive2state[p]
                    for p in primitive2state.keys()
                    if p.type.returns() == arg_type
                ]
                + [(uk, Variable(0, uk))]
            )
        for arg_comb in itertools.product(*args_possibles):
            rules[(primitive, tuple(arg_comb))] = primitive2state[primitive]
    rules[(Variable(0, uk), tuple())] = (uk, Variable(0, uk))
    return DFTA(rules, set())


def __compress__(dsl: DSL, dfta: DFTA[Tuple[Type, DerivableProgram], DerivableProgram]):
    dfta.__remove_unreachable__()
    # Check if we can generalise something
    # states = dfta.states
    # all_prims = set(dsl.list_primitives)
    # for primitive in dsl.list_primitives:
    #     allowed = [set() for _ in primitive.type.arguments()]
    #     for state in dfta.rules.keys():
    #         if state[0] == primitive:
    #             for i, arg in enumerate(state[1]):
    #                 allowed[i].add(arg[1])
    #     generalisable = []

    # pass


def add_commutativity_constraint(
    program: Program, dfta: DFTA[Tuple[Type, DerivableProgram], DerivableProgram]
) -> bool:
    assert program.depth() == 2, f"{program}: depth={program.depth()}"
    assert isinstance(program, Function), f"{program}: type={type(program)}"
    swapped_indices = []
    for i, arg in enumerate(program.arguments):
        assert isinstance(arg, Variable)
        if i != arg.variable:
            swapped_indices.append(i)
    if len(swapped_indices) > 2:
        return False
    x, y = min(swapped_indices), max(swapped_indices)
    fun = program.function
    relevant = []
    for state in dfta.rules:
        if state[0] == fun:
            x_arg = state[1][x]
            y_arg = state[1][y]
            if hash(x_arg[1]) < hash(y_arg[1]):
                relevant.append(state)
    for state in relevant:
        del dfta.rules[state]
    return True


def __simple_constraint(
    program: Program, dfta: DFTA[Tuple[Type, DerivableProgram], DerivableProgram]
) -> bool:
    vars = 0
    for p in program.depth_first_iter():
        if isinstance(p, Variable):
            vars += 1
    if vars > 1:
        return False
    fun = program.function
    relevant = []
    for state in dfta.rules:
        if state[0] == fun:
            success = True
            for arg, sarg in zip(program.arguments, state[1]):
                if isinstance(arg, Function) and arg.function != sarg[1]:
                    success = False
                    break
            if success:
                relevant.append(state)
    for state in relevant:
        del dfta.rules[state]
    return True


def program_to_constraints(
    program: Program, dfta: DFTA[Tuple[Type, DerivableProgram], DerivableProgram]
) -> bool:
    if program.depth() > 3 or not isinstance(program, Function):
        return False
    return __simple_constraint(program, dfta)


def class_to_constaints(
    programs: List[Program],
    dfta: DFTA[Tuple[Type, DerivableProgram], DerivableProgram],
) -> int:
    # Find representative
    representative = programs[0]
    for p in programs:
        if p.size() < representative.size() or (
            p.size() == representative.size() and p.depth() < representative.depth()
        ):
            representative = p
    # Remove representative
    programs.remove(representative)
    # Add constraints
    added = 0
    for p in programs:
        added += program_to_constraints(p, dfta)
    return added


def equivalence_classes_to_filters(
    commutatives: List[Program],
    eq_classes: List[Set[Program]],
    dsl: DSL,
    progress: bool = True,
) -> Tuple[DFTA[Tuple[Type, DerivableProgram], DerivableProgram], Dict[str, float]]:
    added = 0
    pbar = tqdm.tqdm(eq_classes) if progress else eq_classes
    total = 0
    dfta = dsl_2_dfta(dsl)
    initial_size = dfta.size()
    for program in commutatives:
        added += add_commutativity_constraint(program, dfta)
        total += 1
    for eq_class in pbar:
        total += len(eq_class)
        added += class_to_constaints(list(eq_class), dfta)
        if progress:
            pbar.set_postfix_str(
                f"{F.CYAN}{added}{F.RESET}/{total} constraints ({F.GREEN}{added/total:.1%}{F.RESET})"
            )
    __compress__(dsl, dfta)
    return dfta, {
        "initial_size": initial_size,
        "final_size": dfta.size(),
        "added": added,
        "total": total,
    }


def get_filter(
    dfta: DFTA[Tuple[Type, DerivableProgram], DerivableProgram],
    type_request: Type,
    constant_types: Set[Type],
) -> DFTAFilter:
    import copy

    r = copy.deepcopy(dfta.rules)
    dst = r[(Variable(0, uk), tuple())]
    for i, arg_type in enumerate(type_request.arguments()):
        r[(Variable(i, arg_type), tuple())] = dst
    for cst_type in constant_types:
        r[(Constant(cst_type), tuple())] = dst
    return DFTAFilter(DFTA(r, set()))


def type_to_code(type: Type) -> str:
    if isinstance(type, UnknownType):
        return "UnknownType()"
    return f'auto_type("{type}")'


def derivable_program_to_code(
    program: DerivableProgram, type2index: Optional[Dict[Type, int]] = None
) -> str:
    type_part = (
        type_to_code(program.type)
        if type2index is None
        else f"__types[{type2index[program.type]}]"
    )
    if isinstance(program, Primitive):
        return f'Primitive("{program.primitive}", {type_part})'
    elif isinstance(program, Variable):
        return f"Variable({program.variable}, {type_part})"
    assert False, "not implemented"


def dfta_to_code(
    dfta: DFTA[Tuple[Type, DerivableProgram], DerivableProgram], commented: bool = False
) -> str:
    states = list(set(dfta.rules.values()))
    state2index = {s: i for i, s in enumerate(states)}
    letters = list(dfta.alphabet)
    prim2index = {p: i for i, p in enumerate(letters)}
    types_list = list(set(p.type for p in letters).union(set(s[0] for s in states)))
    type2index = {p: i for i, p in enumerate(types_list)}

    def state2code(x: Tuple[Type, DerivableProgram], compressed: bool = False) -> str:
        if compressed:
            return f"__states[{state2index[x]}]"
        return f"(__types[{type2index[x[0]]}], __primitives[{prim2index[x[1]]}])"

    out = ""
    out += "from synth.syntax import Type, Primitive, Variable, Constant, Program, auto_type, UnknownType, DFTA\n"
    out += "from synth.filter import DFTAFilter, Filter\n"
    out += "from typing import Set\n\n"
    out += "__types = [" + ",".join(map(type_to_code, types_list)) + "]\n"
    out += (
        "__primitives = ["
        + ",".join(map(lambda l: derivable_program_to_code(l, type2index), letters))
        + "]\n"
    )
    out += "__states = [" + ",".join(map(state2code, states)) + "]\n"
    out += "__rules = {\n"
    for state, dst in dfta.rules.items():
        dst_code = state2code(dst, True)
        state_code = f"(__primitives[{prim2index[state[0]]}], "
        if len(state[1]) > 0:
            if len(state[1]) == 1:
                state_code += "(" + state2code(state[1][0], True) + ",)"
            else:
                state_code += (
                    "("
                    + ",".join(map(lambda p: state2code(p, True), list(state[1])))
                    + ")"
                )
        else:
            state_code += "tuple()"
        out += f"\t{state_code}): {dst_code},\n"
        if commented:
            out += f"#\t{state} -> {dst}\n"

    out += "}\n\n"
    out += "def get_filter(type_request: Type, constant_types: Set[Type]) -> Filter[Program]:\n"
    out += "\timport copy\n"
    out += "\tr = copy.deepcopy(__rules)\n"
    out += "\tfor i, arg_type in enumerate(type_request.arguments()):\n"
    out += f"\t\tr[(Variable(i, arg_type), tuple())] = __states[{state2index[(uk, Variable(0, uk))]}]\n"
    out += "\tfor cst_type in constant_types:\n"
    out += f"\t\tr[(Constant(cst_type), tuple())] = __states[{state2index[(uk, Variable(0, uk))]}]\n"
    out += "\treturn DFTAFilter(DFTA(r, set()))"
    return out


if __name__ == "__main__":
    import argparse
    import json
    import dsl_loader

    parser = argparse.ArgumentParser(
        description="Transform a JSON file of equivalence classes into constraints"
    )
    dsl_loader.add_dsl_choice_arg(parser)
    parser.add_argument(
        "data",
        type=str,
        help="JSON file containing the equivalence classes",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="dfta_filter_{dsl}.py",
        help="Output python file containing the filter",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="verbose mode",
    )
    parser.add_argument(
        "-c",
        "--comment",
        action="store_true",
        default=False,
        help="comment the output automaton",
    )
    parameters = parser.parse_args()
    data_file: str = parameters.data
    verbose: bool = parameters.verbose
    comment: bool = parameters.comment
    dsl_module = dsl_loader.load_DSL(parameters.dsl)
    output_file: str = parameters.output.format(dsl=parameters.dsl)

    dsl: DSL = dsl_module.dsl

    with open(data_file) as fd:
        dico = json.load(fd)
        classes = dico["classes"]
        commutatives = dico["commutatives"]
        classes = [
            list(map(lambda p: dsl.auto_parse_program(p), eq_class))
            for eq_class in classes
        ]
    if verbose:
        print(f"found {F.CYAN}{len(classes)}{F.RESET} equivalence classes")
    dfta, stats = equivalence_classes_to_filters(commutatives, classes, dsl)
    if verbose:
        print(
            f"found {F.CYAN}{stats['added']}{F.RESET} ({F.GREEN}{stats['added']/stats['total']:.1%}{F.RESET}) constraints"
        )
        print(
            f"reduced size to {F.GREEN}{stats['final_size']/stats['initial_size']:.1%}{F.RESET} of original size"
        )
    with open(output_file, "w") as fd:
        fd.write(dfta_to_code(dfta, commented=comment))
    print(f"Saved to {output_file}!")
