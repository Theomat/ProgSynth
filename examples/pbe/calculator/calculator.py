from synth.semantic import DSLEvaluator
from synth.syntax import DSL, INT, Arrow, PolymorphicType
import numpy as np

from synth.syntax.type_system import PrimitiveType

# a type representing either an int or a float
type = PolymorphicType("int/float")
FLOAT = PrimitiveType("float")

__semantics = {
    "+": lambda a: lambda b: round(a+b,1),
    "-": lambda a: lambda b: round(a-b,1),
    "int2float": lambda a: float(a),
    "1": 1,
    "2": 2,
    "3": 3,
    "3.0": 3.0
}

__primitive_types = {
    # int|float -> int|float -> int|float
    "+": Arrow(type, Arrow(type, type)),
    "-": Arrow(type, Arrow(type, type)),
    "int2float": Arrow(INT, FLOAT),
    "1": INT,
    "2": INT,
    "3": INT,
}

#Short example of a forbidden patterns (if add1 and sub1 are defined in _semantics and _primitive_types)
_forbidden_patterns = [
#    ["add1", "sub1"],
#    ["sub1", "add1"]
]

dsl = DSL(__primitive_types, forbidden_patterns=_forbidden_patterns)
evaluator = DSLEvaluator(__semantics)
lexicon = [round(x,1) for x in np.arange(-256, 256+1, 0.1)]
