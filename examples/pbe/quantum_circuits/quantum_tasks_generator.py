from typing import (
    Any,
    Dict,
    Set,
    Tuple,
    TypeVar,
)
from itertools import product

import numpy as np

from synth.syntax import (
    List,
    DSL,
    auto_type,
    Program,
    CFG,
    UCFG,
    Type,
    ProbUGrammar,
    enumerate_prob_u_grammar,
    DFTA,
)
from synth.semantic import DSLEvaluator
from synth.syntax.grammars.det_grammar import DerivableProgram
from synth import Task, PBE, Dataset

import tqdm

from quantum import QiskitTester

import qiskit as qk
from qiskit.transpiler.passes import SolovayKitaev


__syntax = auto_type(
    {
        "H": "circuit -> int -> circuit",
        "T": "circuit -> int -> circuit",
        "Tdg": "circuit -> int -> circuit",
        "CNOT": "circuit -> int -> int -> circuit",
        "I": "circuit -> int -> circuit",
        "S": "circuit -> int -> circuit",
        "X": "circuit -> int -> circuit",
        "Y": "circuit -> int -> circuit",
        "Z": "circuit -> int -> circuit",
        "SX": "circuit -> int -> circuit",
        "SXdg": "circuit -> int -> circuit",
        "CY": "circuit -> int -> int -> circuit",
        "CZ": "circuit -> int -> int -> circuit",
        "CS": "circuit -> int -> int -> circuit",
        "CH": "circuit -> int -> int -> circuit",
        "SWAP": "circuit -> int -> int -> circuit",
        "iSWAP": "circuit -> int -> int -> circuit",
    }
)

# Trick to always return QT while running statement X:
# QT if X is not None else QT
__semantics = {
    "H": lambda QT: lambda q1: QT if QT.circuit.h(QT.q(q1)) is not None else QT,
    "T": lambda QT: lambda q1: QT if QT.circuit.t(QT.q(q1)) is not None else QT,
    "Tdg": lambda QT: lambda q1: QT if QT.circuit.tdg(QT.q(q1)) is not None else QT,
    "CNOT": lambda QT: lambda q1: lambda q2: QT
    if QT.circuit.cnot(QT.q(q1), QT.q(q2)) is not None
    else QT,
    "I": lambda QT: lambda q1: QT if QT.circuit.id(QT.q(q1)) is not None else QT,
    "S": lambda QT: lambda q1: QT if QT.circuit.s(QT.q(q1)) is not None else QT,
    "X": lambda QT: lambda q1: QT if QT.circuit.x(QT.q(q1)) is not None else QT,
    "Y": lambda QT: lambda q1: QT if QT.circuit.y(QT.q(q1)) is not None else QT,
    "Z": lambda QT: lambda q1: QT if QT.circuit.z(QT.q(q1)) is not None else QT,
    "SX": lambda QT: lambda q1: QT if QT.circuit.sx(QT.q(q1)) is not None else QT,
    "SXdg": lambda QT: lambda q1: QT if QT.circuit.sxdg(QT.q(q1)) is not None else QT,
    "CY": lambda QT: lambda q1: lambda q2: QT
    if QT.circuit.cy(QT.q(q1), QT.q(q2)) is not None
    else QT,
    "CZ": lambda QT: lambda q1: lambda q2: QT
    if QT.circuit.cz(QT.q(q1), QT.q(q2)) is not None
    else QT,
    "CS": lambda QT: lambda q1: lambda q2: QT
    if QT.circuit.append(qk.circuit.library.SGate().control(1), (QT.q(q1), QT.q(q2)))
    is not None
    else QT,
    "CH": lambda QT: lambda q1: lambda q2: QT
    if QT.circuit.ch(QT.q(q1), QT.q(q2)) is not None
    else QT,
    "SWAP": lambda QT: lambda q1: lambda q2: QT
    if QT.circuit.swap(QT.q(q1), QT.q(q2)) is not None
    else QT,
    "iSWAP": lambda QT: lambda q1: lambda q2: QT
    if QT.circuit.iswap(QT.q(q1), QT.q(q2)) is not None
    else QT,
}

# Is this important?
with QiskitTester(1) as QT:
    QT.circuit.t(0)
    QT.circuit.t(0)
qk.circuit.equivalence_library.StandardEquivalenceLibrary.add_equivalence(
    qk.circuit.library.SGate(), QT.circuit
)

with QiskitTester(1) as QT:
    QT.circuit.tdg(0)
    QT.circuit.tdg(0)
qk.circuit.equivalence_library.StandardEquivalenceLibrary.add_equivalence(
    qk.circuit.library.SdgGate(), QT.circuit
)


class ParametricSubstitution(qk.transpiler.TransformationPass):
    def run(self, dag):
        # iterate over all operations
        for node in dag.op_nodes():
            # print(node.op.name, node.op.params)
            # if we hit a RYY or RZZ gate replace it

            if node.op.name in ["cp"]:
                replacement = qk.QuantumCircuit(2)
                replacement.p(node.op.params[0] / 2, 0)
                replacement.cx(0, 1)
                replacement.p(-node.op.params[0] / 2, 1)
                replacement.cx(0, 1)
                replacement.p(node.op.params[0] / 2, 1)

                # replace the node with our new decomposition
                dag.substitute_node_with_dag(
                    node, qk.converters.circuit_to_dag(replacement)
                )

            if node.op.name in ["p"] and node.op.params[0] == np.pi / 2:

                # calculate the replacement
                replacement = qk.QuantumCircuit(1)
                replacement.s([0])

                # replace the node with our new decomposition
                dag.substitute_node_with_dag(
                    node, qk.converters.circuit_to_dag(replacement)
                )

            elif node.op.name in ["p"] and node.op.params[0] == 3 * np.pi / 2:

                # calculate the replacement
                replacement = qk.QuantumCircuit(1)
                replacement.tdg([0])
                replacement.tdg([0])

                # replace the node with our new decomposition
                dag.substitute_node_with_dag(
                    node, qk.converters.circuit_to_dag(replacement)
                )

            elif node.op.name in ["p"] and node.op.params[0] == 5 * np.pi / 2:

                # calculate the replacement
                replacement = qk.QuantumCircuit(1)
                replacement.t([0])
                replacement.t([0])

                # replace the node with our new decomposition
                dag.substitute_node_with_dag(
                    node, qk.converters.circuit_to_dag(replacement)
                )

        return dag


def decompose(
    circuit: qk.QuantumCircuit,
    backend: qk.AerWrapper,
    pm: qk.transpiler.PassManager,
    skd: SolovayKitaev,
) -> qk.QuantumCircuit:

    transpiled = qk.transpile(circuit, backend)
    circuit2 = pm.run(transpiled)
    discretized = skd(circuit2)
    return qk.transpile(discretized, backend, ["h", "cx", "t", "tdg"])


def circuit_to_program(circuit: qk.QuantumCircuit, dsl: DSL, tr: Type) -> Program:

    program = "var0"
    for inst in circuit.data:
        name: str = inst.operation.name
        program = f"({name.capitalize()} {program}"
        for qbit in inst.qubits:
            index, registers = circuit.find_bit(qbit)
            register, idx_in_reg = registers[0]
            program += f" {register.size-index - 1}"
        program += ")"
    return dsl.parse_program(program.replace("Cx", "CNOT"), tr)


class PartialQuantumCircuitEvaluator(DSLEvaluator):
    def __init__(self, semantics: Dict[str, Any], nqbits: int = 3) -> None:
        super().__init__(semantics, False)
        self.nqbits = nqbits
        self.backend = qk.Aer.get_backend("unitary_simulator")

    def eval(self, program: Program, input: List) -> Any:
        with QiskitTester(self.nqbits) as QT:
            super().eval(program, [QT] + input)

            return QT.circuit


U = TypeVar("U")
V = TypeVar("V")


def __cfg2dfta__(
    grammar: CFG,
) -> DFTA[Tuple[Type, int], DerivableProgram]:
    StateT = Tuple[Type, int]
    dfta_rules: Dict[Tuple[DerivableProgram, Tuple[StateT, ...]], StateT] = {}
    max_depth = grammar.max_program_depth()
    all_cases: Dict[
        Tuple[int, Tuple[Type, ...]], Set[Tuple[Tuple[Type, int], ...]]
    ] = {}
    for S in grammar.rules:
        for P in grammar.rules[S]:
            args = grammar.rules[S][P][0]
            if len(args) == 0:
                dfta_rules[(P, ())] = (P.type, 0)
            else:
                key = (len(args), tuple([arg[0] for arg in args]))
                if key not in all_cases:
                    all_cases[key] = set(
                        [
                            tuple(x)
                            for x in product(
                                *[
                                    [(arg[0], j) for j in range(max_depth)]
                                    for arg in args
                                ]
                            )
                        ]
                    )
                for nargs in all_cases[key]:
                    new_depth = max(i for _, i in nargs) + 1
                    if new_depth >= max_depth:
                        continue
                    dfta_rules[(P, nargs)] = (
                        S[0],
                        new_depth,
                    )
    r = grammar.type_request.returns()
    dfta = DFTA(dfta_rules, {(r, x) for x in range(max_depth)})
    dfta.reduce()
    return dfta


def __generate_syntax__(
    nqbits: int, max_operations: int, verbose: bool
) -> Tuple[PartialQuantumCircuitEvaluator, DSL, ProbUGrammar]:
    tr = auto_type("circuit -> circuit")
    type_int = auto_type("int")
    if verbose:
        print("Augmenting DSL...", end="")
    # Make copies to have no side-effects
    syntax = {x: y for x, y in __syntax.items()}
    semantics = {x: y for x, y in __semantics.items()}
    # Add constants for qbits
    for n in range(nqbits):
        syntax[str(n)] = type_int
        semantics[str(n)] = n
    # DSL + Evaluator
    dsl = DSL(syntax)
    evaluator = PartialQuantumCircuitEvaluator(semantics, nqbits)
    if verbose:
        print("done!\nGenerating grammar...", end="")
    # PCFG
    cfg = CFG.depth_constraint(dsl, tr, max_operations)

    dfta = __cfg2dfta__(cfg)
    if verbose:
        print("done!\nFixing grammar...", end="")
    depths = {y for x, y in dfta.states if x == type_int}
    for depth in depths:
        for n in range(nqbits):
            s = (auto_type("int" + str(n)), depth)
            dfta.states.add(s)
            dfta.rules[(dsl.get_primitive(str(n)), ())] = s
    for (P, args), dst in list(dfta.rules.items()):
        if len(args) == 3:
            del dfta.rules[(P, args)]
            for n1 in range(nqbits):
                for n2 in range(nqbits):
                    if n1 == n2:
                        continue
                    n_args = (
                        args[0],
                        (auto_type("int" + str(n1)), args[1][1]),
                        (auto_type("int" + str(n2)), args[2][1]),
                    )
                    dfta.rules[(P, n_args)] = dst
        elif len(args) == 2:
            del dfta.rules[(P, args)]
            for n1 in range(nqbits):
                n_args = (
                    args[0],
                    (auto_type("int" + str(n1)), args[1][1]),
                )
                dfta.rules[(P, n_args)] = dst
    if verbose:
        print("done!\nConverting to probabilistic grammar...", end="")
    pcfg = ProbUGrammar.uniform(UCFG.from_DFTA_with_ngrams(dfta, 2))
    if verbose:
        print("done!")
    return evaluator, dsl, pcfg


def generate_tasks(
    nqbits: int = 3, n_tasks: int = 1000, max_operations: int = 5, verbose: bool = False
) -> Dataset[PBE]:
    evaluator, dsl, pcfg = __generate_syntax__(nqbits, max_operations, verbose)
    tr = pcfg.type_request
    if verbose:
        print("Loading quantum backend...", end="")

    backend = qk.Aer.get_backend("unitary_simulator")
    skd = SolovayKitaev()
    pm = qk.transpiler.PassManager()
    pm.append(ParametricSubstitution())
    if verbose:
        print("done!")

    tasks = []
    pbar = None
    if verbose:
        pbar = tqdm.tqdm(total=n_tasks, desc="Task Generation")
    for program in enumerate_prob_u_grammar(pcfg):
        # print("Evaluating:", str(program))
        complex_circuit = evaluator.eval(program, [])
        try:
            base_circuit = decompose(complex_circuit, backend, pm, skd)
        except qk.transpiler.exceptions.TranspilerError:
            continue
        task = Task[PBE](tr, PBE([]), circuit_to_program(base_circuit, dsl, tr))
        tasks.append(task)
        if pbar:
            pbar.update(1)
        if len(tasks) >= n_tasks:
            break
    if pbar:
        pbar.close()
    return Dataset(tasks, {"generated:": True})


if __name__ == "__main__":
    for task in generate_tasks(2, 100, 5, verbose=True):
        print("Task:", task)
