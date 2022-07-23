"""
Module that contains anything relevant to pruning
"""
from synth.pruning.pruner import Pruner, UnionPruner
from synth.pruning.syntactic_pruner import (
    UseAllVariablesPruner,
    FunctionPruner,
    SyntacticPruner,
    SetPruner,
)
from synth.pruning.type_constraints import (
    export_syntax_to_python,
    produce_new_syntax_for_constraints,
    produce_new_syntax_for_sketch,
)
