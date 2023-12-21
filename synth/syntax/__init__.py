"""
Module that contains anything relevant to the syntax
"""
from synth.syntax.dsl import DSL
from synth.syntax.program import Primitive, Variable, Function, Lambda, Program
from synth.syntax.type_helper import guess_type, FunctionType, auto_type
from synth.syntax.type_system import (
    Type,
    match,
    PrimitiveType,
    PolymorphicType,
    FixedPolymorphicType,
    Generic,
    TypeFunctor,
    GenericFunctor,
    List,
    Arrow,
    Sum,
    UnknownType,
    INT,
    BOOL,
    STRING,
    UNIT,
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
    ProgramEnumerator,
    bs_enumerate_prob_grammar,
    bps_enumerate_prob_grammar,
    hs_enumerate_prob_grammar,
    hs_enumerate_prob_u_grammar,
    hs_enumerate_bucket_prob_grammar,
    hs_enumerate_bucket_prob_u_grammar,
    split,
)
