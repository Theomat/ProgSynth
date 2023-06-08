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
)
from synth.semantic import DSLEvaluator


import qiskit as qk

__syntax = auto_type(
    {
        "H": "circuit -> int -> circuit",
        "T": "circuit -> int -> circuit",
        "Tdg": "circuit -> int -> circuit",
        "CNOT": "circuit -> int -> int -> circuit",
    }
)


__semantics = {
    "H": lambda QT, q1: QT.circuit.h(QT.q(q1)),
    "T": lambda QT, q1: QT.circuit.t(QT.q(q1)),
    "Tdg": lambda QT, q1: QT.circuit.tdg(QT.q(q1)),
    "CNOT": lambda QT, q1, q2: QT.circuit.cnot(QT.q(q1), QT.q(q2)),
}


class QuantumCircuitEvaluator(DSLEvaluator):
    def __init__(
        self, semantics: Dict[str, Any], nqbits: int = 3, use_cache: bool = True
    ) -> None:
        super().__init__(semantics, use_cache)
        self.nqbits = nqbits
        self.backend = qk.Aer.get_backend("unitary_simulator")

    def eval(self, program: Program, input: List) -> Any:
        reg_q = qk.QuantumRegister(self.nqbits, "q")
        circuit = qk.QuantumCircuit(reg_q)
        new_circuit = super().eval(program, [circuit] + input)
        job = self.backend.run(new_circuit)
        result = job.result()
        outputstate = result.get_unitary(new_circuit, decimals=3)
        print(outputstate)


dsl = DSL(__syntax)
evaluator = QuantumCircuitEvaluator(__semantics)
