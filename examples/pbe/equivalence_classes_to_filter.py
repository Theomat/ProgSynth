from typing import List, Dict, Optional, Set, Tuple
import itertools
import copy

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
from synth.filter import DFTAFilter, LocalStatelessFilter, Filter
from synth.syntax.grammars.grammar import DerivableProgram

uk = UnknownType()


class FiltersBuilder:
    def __init__(
        self, dfta: DFTA[Tuple[Type, DerivableProgram], DerivableProgram]
    ) -> None:
        self.dfta = dfta
        self.equal_parameters_reject: Set[
            Tuple[DerivableProgram, Tuple[Tuple[int, int], ...]]
        ] = set()
        self.stats = {
            "dfta.size.initial": self.dfta.size(),
            "dfta.size.final": self.dfta.size(),
            "constraints.total": 0,
            "constraints.successes": 0,
        }

    def add_commutativity_constraint(self, program: Program) -> bool:
        assert program.depth() == 2, f"{program}: depth={program.depth()}"
        assert isinstance(program, Function), f"{program}: type={type(program)}"
        self.stats["constraints.total"] += 1
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
        for state in self.dfta.rules:
            if state[0] == fun:
                x_arg = state[1][x]
                y_arg = state[1][y]
                if hash(x_arg[1]) < hash(y_arg[1]):
                    relevant.append(state)
        for state in relevant:
            del self.dfta.rules[state]
        self.stats["constraints.successes"] += 1
        return True

    def __simple_constraint(self, program: Program) -> bool:
        vars = 0
        for p in program.depth_first_iter():
            if isinstance(p, Variable):
                vars += 1
        if vars > 1:
            return False
        fun = program.function
        relevant = []
        for state in self.dfta.rules:
            if state[0] == fun:
                success = True
                for arg, sarg in zip(program.arguments, state[1]):
                    if isinstance(arg, Function) and arg.function != sarg[1]:
                        success = False
                        break
                if success:
                    relevant.append(state)
        for state in relevant:
            del self.dfta.rules[state]
        return True

    def __program_to_stateless_constraint(self, program: Program) -> bool:
        if program.depth() == 2 and isinstance(program, Function):
            diff = []
            for i, arg in enumerate(program.arguments):
                if not isinstance(arg, Variable):
                    return False
                if i != arg.variable:
                    diff.append((min(i, arg.variable), max(i, arg.variable)))
            self.equal_parameters_reject.add((program.function, tuple(diff)))
            return True
        return False

    def forbid_program(self, program: Program) -> bool:
        self.stats["constraints.total"] += 1
        if program.depth() > 3 or not isinstance(program, Function):
            return False
        out = self.__simple_constraint(
            program
        ) or self.__program_to_stateless_constraint(program)
        if out:
            self.stats["constraints.successes"] += 1
        return out

    def add_equivalence_class(self, programs: List[Program]) -> int:
        # Find representative
        representative = programs[0]
        for p in programs:
            if p.size() < representative.size() or (
                p.size() == representative.size() and p.depth() < representative.depth()
            ):
                representative = p
        # Remove representative
        eq_class = [p for p in programs if p != representative]
        added = 0
        for p in eq_class:
            added += self.forbid_program(p)
        return added

    def compress(self):
        self.dfta.__remove_unreachable__()
        self.stats["dfta.size.final"] = self.dfta.size()

    def get_filter(
        self,
        type_request: Type,
        constant_types: Set[Type],
    ) -> Filter[Program]:
        # DFTA part
        r = copy.deepcopy(self.dfta.rules)
        dst = r[(Variable(0, uk), tuple())]
        for i, arg_type in enumerate(type_request.arguments()):
            r[(Variable(i, arg_type), tuple())] = dst
        for cst_type in constant_types:
            r[(Constant(cst_type), tuple())] = dst
        x = DFTAFilter(DFTA(r, set()))

        if len(self.equal_parameters_reject) == 0:
            return x
        # Local Stateless Part
        def make_equal_filter(to_look):
            return lambda *args: all(args[i] == args[j] for i, j in to_look)

        def make_or(f, g):
            return lambda *args: g(*args) or f(*args)

        should_reject = {}
        for p, to_look in self.equal_parameters_reject:
            f = make_equal_filter(to_look)
            key = p.primitive
            if key in should_reject:
                old = should_reject[key]
                should_reject[key] = make_or(old, f)
            else:
                should_reject[key] = f
        filter = LocalStatelessFilter(should_reject)
        return filter.intersection(x)

    def to_code(self, commented: bool = False) -> str:
        states = list(set(self.dfta.rules.values()))
        state2index = {s: i for i, s in enumerate(states)}
        letters = list(self.dfta.alphabet)
        prim2index = {p: i for i, p in enumerate(letters)}
        types_list = list(set(p.type for p in letters).union(set(s[0] for s in states)))
        type2index = {p: i for i, p in enumerate(types_list)}

        def state2code(
            x: Tuple[Type, DerivableProgram], compressed: bool = False
        ) -> str:
            if compressed:
                return f"__states[{state2index[x]}]"
            return f"(__types[{type2index[x[0]]}], __primitives[{prim2index[x[1]]}])"

        out = ""
        out += "from synth.syntax import Type, Primitive, Variable, Constant, Program, auto_type, UnknownType, DFTA\n"
        out += "from synth.filter import DFTAFilter, Filter, LocalStatelessFilter\n"
        out += "from typing import Set\n\n"
        # DFTA PART
        out += "__types = [" + ",".join(map(type_to_code, types_list)) + "]\n"
        out += (
            "__primitives = ["
            + ",".join(map(lambda l: derivable_program_to_code(l, type2index), letters))
            + "]\n"
        )
        out += "__states = [" + ",".join(map(state2code, states)) + "]\n"
        out += "__rules = {\n"
        for state, dst in self.dfta.rules.items():
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
        # LOCAL STATELESS PART
        if len(self.equal_parameters_reject) > 0:
            out += "__should_reject = {\n"
            # build dict
            should_reject = {}
            for p, to_look in self.equal_parameters_reject:
                key = p.primitive
                code = f"all(args[i] == args[j] for i,j in {to_look})"
                if key in should_reject:
                    old = should_reject[key]
                    should_reject[key] = old + " or " + code
                else:
                    should_reject[key] = code
            # print it
            for p, code in should_reject.items():
                out += f'\t"{p}": lambda *args: {code},\n'
            out += "}\n\n"
        # GETTER FUNCTION
        out += "def get_filter(type_request: Type, constant_types: Set[Type]) -> Filter[Program]:\n"
        out += "\timport copy\n"
        out += "\tr = copy.deepcopy(__rules)\n"
        out += "\tfor i, arg_type in enumerate(type_request.arguments()):\n"
        out += f"\t\tr[(Variable(i, arg_type), tuple())] = __states[{state2index[(uk, Variable(0, uk))]}]\n"
        out += "\tfor cst_type in constant_types:\n"
        out += f"\t\tr[(Constant(cst_type), tuple())] = __states[{state2index[(uk, Variable(0, uk))]}]\n"
        out += "\tx: Filter[Program] = DFTAFilter(DFTA(r, set()))\n"
        if len(self.equal_parameters_reject) > 0:
            out += "\ty = LocalStatelessFilter(__should_reject)\n"
            out += "\tx = x.intersection(y)\n"
        out += "\treturn x\n"
        return out

    @staticmethod
    def from_dsl(dsl: DSL) -> "FiltersBuilder":
        primitives = dsl.list_primitives
        primitive2state: Dict[Primitive, Tuple[Type, Primitive]] = {}
        rules: Dict[
            Tuple[DerivableProgram, Tuple[Tuple[Type, DerivableProgram], ...]],
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
        return FiltersBuilder(DFTA(rules, set()))


def equivalence_classes_to_filters(
    commutatives: List[Program],
    eq_classes: List[Set[Program]],
    dsl: DSL,
    progress: bool = True,
) -> FiltersBuilder:
    added = 0
    pbar = tqdm.tqdm(eq_classes) if progress else eq_classes
    total = 0
    builder = FiltersBuilder.from_dsl(dsl)
    for program in commutatives:
        added += builder.add_commutativity_constraint(program)
        total += 1
    for eq_class in pbar:
        total += len(eq_class) - 1
        this_class = list(eq_class)
        added += builder.add_equivalence_class(this_class)
        if progress:
            pbar.set_postfix_str(
                f"{F.CYAN}{added}{F.RESET}/{total} constraints ({F.GREEN}{added/total:.1%}{F.RESET})"
            )
    builder.compress()

    return builder


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
        commutatives = list(
            map(lambda p: dsl.auto_parse_program(p), dico["commutatives"])
        )
        classes = [
            list(map(lambda p: dsl.auto_parse_program(p), eq_class))
            for eq_class in classes
        ]
    if verbose:
        print(f"found {F.CYAN}{len(classes)}{F.RESET} equivalence classes")
    builder = equivalence_classes_to_filters(commutatives, classes, dsl)
    if verbose:
        stats = builder.stats
        print(
            f"found {F.CYAN}{stats['constraints.successes']}{F.RESET} ({F.GREEN}{stats['constraints.successes'] / stats['constraints.total']:.1%}{F.RESET}) constraints"
        )
        print(
            f"reduced size to {F.GREEN}{stats['dfta.size.final']/stats['dfta.size.initial']:.1%}{F.RESET} of original size"
        )
    with open(output_file, "w") as fd:
        fd.write(builder.to_code(commented=comment))
    print(f"Saved to {output_file}!")
