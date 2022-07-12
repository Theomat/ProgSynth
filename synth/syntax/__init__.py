"""
Module that contains anything relevant to the syntax
"""
from synth.syntax.dsl import DSL
from synth.syntax.program import Primitive, Variable, Function, Lambda, Program
from synth.syntax.type_system import (
    Type,
    FunctionType,
    guess_type,
    match,
    PrimitiveType,
    PolymorphicType,
    List,
    Arrow,
    INT,
    BOOL,
    STRING,
)
from synth.syntax.grammars import (
    CFG,
    # UPCFG,
    TTCFG,
    Grammar,
    DetGrammar,
    enumerate_pcfg,
    enumerate_bucket_pcfg,
    # split,
)
