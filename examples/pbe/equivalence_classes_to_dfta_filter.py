from typing import List, Dict, Optional, Tuple
import itertools

from colorama import Fore as F

from synth.syntax import Program, Primitive, Variable, Function, DSL, Type
from synth.syntax import DFTA

import tqdm

from synth.syntax.grammars.grammar import DerivableProgram
from synth.syntax.type_system import UnknownType

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


def program_to_constraints(
    program: Program, dfta: DFTA[Tuple[Type, DerivableProgram], DerivableProgram]
) -> bool:
    if program.depth() > 3 or not isinstance(program, Function):
        return False
    fun = program.function
    relevant = []
    for state, dst in dfta.rules.items():
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


def class_to_constaints(
    eq_class: List[str],
    dfta: DFTA[Tuple[Type, DerivableProgram], DerivableProgram],
    dsl: DSL,
) -> int:
    programs = list(map(lambda p: dsl.auto_parse_program(p), eq_class))
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


def dfta_to_code(dfta: DFTA[Tuple[Type, DerivableProgram], DerivableProgram]) -> str:
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
        elems = map(lambda p: state2code(p, True), list(state[1]))
        state_code = (
            f"(__primitives[{prim2index[state[0]]}], tuple(" + ",".join(elems) + "))"
        )
        out += f"\t{state_code}: {dst_code},\n"
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
    parameters = parser.parse_args()
    data_file: str = parameters.data
    verbose: bool = parameters.verbose
    dsl_module = dsl_loader.load_DSL(parameters.dsl)
    output_file: str = parameters.output.format(dsl=parameters.dsl)

    dfta = dsl_2_dfta(dsl_module.dsl)
    initial_size = dfta.size()

    with open(data_file) as fd:
        classes = json.load(fd)
    if verbose:
        print(f"found {F.CYAN}{len(classes)}{F.RESET} equivalence classes")
    added = 0
    pbar = tqdm.tqdm(classes)
    total = 0
    for eq_class in pbar:
        total += len(eq_class)
        added += class_to_constaints(eq_class, dfta, dsl_module.dsl)
        pbar.set_postfix_str(
            f"{F.CYAN}{added}{F.RESET}/{total} constraints ({F.GREEN}{added/total:.1%}{F.RESET})"
        )
    __compress__(dsl_module.dsl, dfta)
    if verbose:
        print(
            f"found {F.CYAN}{added}{F.RESET} ({F.GREEN}{added/total:.1%}{F.RESET}) constraints"
        )
        print(
            f"reduced size to {F.GREEN}{dfta.size()/initial_size:.1%}{F.RESET} of original size"
        )
    with open(output_file, "w") as fd:
        fd.write(dfta_to_code(dfta))
    print(f"Saved to {output_file}!")
