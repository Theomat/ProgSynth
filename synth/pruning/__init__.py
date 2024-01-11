"""
Module that contains anything relevant to pruning
"""
from synth.pruning.pruner import Pruner, UnionPruner, IntersectionPruner
from synth.pruning.dfta_pruner import DFTAPruner
from synth.pruning.syntactic_pruner import (
    UseAllVariablesPruner,
    FunctionPruner,
    SyntacticPruner,
    SetPruner,
)
from synth.pruning.constraints import add_constraints
