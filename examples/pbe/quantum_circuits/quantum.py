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
from synth.syntax.program import Primitive

__syntax = auto_type(
    {
        "H": "circuit -> int -> circuit",
        "T": "circuit -> int -> circuit",
        "Tdg": "circuit -> int -> circuit",
        "CNOT": "circuit -> int -> int -> circuit",
    }
)


__semantics = {
    "H": lambda QT: lambda q1: QT if QT.circuit.h(QT.q(q1)) is not None else QT,
    "T": lambda QT: lambda q1: QT if QT.circuit.t(QT.q(q1)) is not None else QT,
    "Tdg": lambda QT: lambda q1: QT if QT.circuit.tdg(QT.q(q1)) is not None else QT,
    "CNOT": lambda QT: lambda q1: lambda q2: QT
    if QT.circuit.cnot(QT.q(q1), QT.q(q2)) is not None
    else QT,
}


class QiskitTester:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.unitary_matrix = None
        self.qreg_q = qk.QuantumRegister(self.n_qubits, "q")
        self.circuit = qk.QuantumCircuit(self.qreg_q)

    def q(self, q_num: int) -> int:
        return self.n_qubits - 1 - q_num

    def __enter__(self):
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __str__(self) -> str:
        return self.circuit.__str__()

    def execute(self, backend: qk.AerWrapper) -> np.ndarray:
        return np.array(qk.execute(self.circuit, backend).result().get_unitary()).T


class QuantumCircuitEvaluator(DSLEvaluator):
    def __init__(self, semantics: Dict[Primitive, Any], nqbits: int = 3) -> None:
        super().__init__(semantics, False)
        self.nqbits = nqbits
        self.backend = qk.Aer.get_backend("unitary_simulator")

    def eval(self, program: Program, input: List) -> Any:
        with QiskitTester(self.nqbits) as QT:
            super().eval(program, [QT] + input)

            return QT.execute(self.backend)


dsl = DSL(__syntax)
evaluator = QuantumCircuitEvaluator(dsl.instantiate_semantics(__semantics))
