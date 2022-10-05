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
from synth.syntax.automata import DFA, DFTA
from synth.syntax.grammars import (
    CFG,
    UCFG,
    TTCFG,
    Grammar,
    DetGrammar,
    UGrammar,
    ProbDetGrammar,
    ProbUGrammar,
    TaggedDetGrammar,
    TaggedUGrammar,
    enumerate_prob_grammar,
    enumerate_prob_u_grammar,
    enumerate_bucket_prob_grammar,
    enumerate_bucket_prob_u_grammar,
    # split,
)
