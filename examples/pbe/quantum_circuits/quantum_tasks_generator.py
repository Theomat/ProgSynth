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


__semantics = {
    "H": lambda QT, q1: QT.circuit.h(QT.q(q1)),
    "T": lambda QT, q1: QT.circuit.t(QT.q(q1)),
    "Tdg": lambda QT, q1: QT.circuit.tdg(QT.q(q1)),
    "CNOT": lambda QT, q1, q2: QT.circuit.cnot(QT.q(q1), QT.q(q2)),
    "I": lambda QT, q1: QT.circuit.id(QT.q(q1)),
    "S": lambda QT, q1: QT.circuit.s(QT.q(q1)),
    "X": lambda QT, q1: QT.circuit.x(QT.q(q1)),
    "Y": lambda QT, q1: QT.circuit.y(QT.q(q1)),
    "Z": lambda QT, q1: QT.circuit.z(QT.q(q1)),
    "SX": lambda QT, q1: QT.circuit.sx(QT.q(q1)),
    "SXdg": lambda QT, q1: QT.circuit.sxdg(QT.q(q1)),
    "CY": lambda QT, q1, q2: QT.circuit.cy(QT.q(q1), QT.q(q2)),
    "CZ": lambda QT, q1, q2: QT.circuit.cz(QT.q(q1), QT.q(q2)),
    "CS": lambda QT, q1, q2: QT.circuit.append(
        qk.circuit.library.SGate().control(1), (QT.q(q1), QT.q(q2))
    ),
    "CH": lambda QT, q1, q2: QT.circuit.ch(QT.q(q1), QT.q(q2)),
    "SWAP": lambda QT, q1, q2: QT.circuit.swap(QT.q(q1), QT.q(q2)),
    "iSWAP": lambda QT, q1, q2: QT.circuit.iswap(QT.q(q1), QT.q(q2)),
}

dsl = DSL(__syntax)
evaluator = QuantumCircuitEvaluator(__semantics)


def generate_tasks(n_tasks: int = 1000, max_operations: int = 5) -> Dataset[PBE]:
    tr = auto_type("circuit -> circuit")
    cfg = CFG.depth_constraint(dsl, tr, max_operations)
    pcfg = ProbDetGrammar.uniform(cfg)

    tasks = []
    for program in enumerate_prob_grammar(pcfg):
        task = Task[PBE](tr, PBE(), program)
        tasks.append(task)
        if len(tasks) >= n_tasks:
            break
    return Dataset(tasks, {"generated:": True})
