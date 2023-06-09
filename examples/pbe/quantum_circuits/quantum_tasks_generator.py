from typing import Any, Dict
import numpy as np

from synth.syntax import (
    PolymorphicType,
    Arrow,
    INT,
    BOOL,
    List,
    DSL,
    auto_type,
    Program,
    CFG,
    ProbDetGrammar,
    enumerate_prob_grammar,
)
from synth.semantic import DSLEvaluator
from synth import Task, PBE, Dataset


from quantum import QuantumCircuitEvaluator

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
        "0": "int",
        "1": "int",
        "2": "int",
    }
)


__semantics = {
    "H": lambda QT: lambda q1: QT.circuit.h(QT.q(q1)),
    "T": lambda QT: lambda q1: QT.circuit.t(QT.q(q1)),
    "Tdg": lambda QT: lambda q1: QT.circuit.tdg(QT.q(q1)),
    "CNOT": lambda QT: lambda q1: lambda q2: QT.circuit.cnot(QT.q(q1), QT.q(q2)),
    "I": lambda QT: lambda q1: QT.circuit.id(QT.q(q1)),
    "S": lambda QT: lambda q1: QT.circuit.s(QT.q(q1)),
    "X": lambda QT: lambda q1: QT.circuit.x(QT.q(q1)),
    "Y": lambda QT: lambda q1: QT.circuit.y(QT.q(q1)),
    "Z": lambda QT: lambda q1: QT.circuit.z(QT.q(q1)),
    "SX": lambda QT: lambda q1: QT.circuit.sx(QT.q(q1)),
    "SXdg": lambda QT: lambda q1: QT.circuit.sxdg(QT.q(q1)),
    "CY": lambda QT: lambda q1: lambda q2: QT.circuit.cy(QT.q(q1), QT.q(q2)),
    "CZ": lambda QT: lambda q1: lambda q2: QT.circuit.cz(QT.q(q1), QT.q(q2)),
    "CS": lambda QT: lambda q1: lambda q2: QT.circuit.append(
        qk.circuit.library.SGate().control(1), (QT.q(q1), QT.q(q2))
    ),
    "CH": lambda QT: lambda q1: lambda q2: QT.circuit.ch(QT.q(q1), QT.q(q2)),
    "SWAP": lambda QT: lambda q1: lambda q2: QT.circuit.swap(QT.q(q1), QT.q(q2)),
    "iSWAP": lambda QT: lambda q1: lambda q2: QT.circuit.iswap(QT.q(q1), QT.q(q2)),
}


class ParametricSubstitution(qk.transpiler.TransformationPass):
    def run(self, dag):
        # iterate over all operations
        for node in dag.op_nodes():
            print(node.op.name, node.op.params)
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


def circuit_to_program(circuit: qk.QuantumCircuit) -> Program:
    pass


def generate_tasks(
    nqbits: int = 3, n_tasks: int = 1000, max_operations: int = 5
) -> Dataset[PBE]:
    tr = auto_type("circuit -> circuit")
    # Add constants for qbits
    for n in range(nqbits):
        __syntax[str(n)] = auto_type("int")
        __semantics[str(n)] = n
    # DSL + Evaluator
    dsl = DSL(__syntax)
    evaluator = QuantumCircuitEvaluator(__semantics, nqbits)
    # PCFG
    cfg = CFG.depth_constraint(dsl, tr, max_operations)
    pcfg = ProbDetGrammar.uniform(cfg)

    backend = qk.Aer.get_backend("unitary_simulator")
    skd = SolovayKitaev()
    pm = qk.transpiler.PassManager()
    pm.append(ParametricSubstitution())

    tasks = []
    for program in enumerate_prob_grammar(pcfg):
        name = str(program)
        print("Evaluating:", name)
        complex_circuit = evaluator.eval(program, [])
        base_circuit = decompose(complex_circuit, backend, pm, skd)
        task = Task[PBE](tr, PBE(), circuit_to_program(base_circuit))
        tasks.append(task)
        if len(tasks) >= n_tasks:
            break
    return Dataset(tasks, {"generated:": True})


if __name__ == "__main__":
    for task in generate_tasks(100, 5):
        print(task)
