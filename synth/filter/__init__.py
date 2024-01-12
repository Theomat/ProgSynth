"""
Module that contains anything relevant to pruning
"""
from synth.filter.pruner import Pruner, UnionPruner, IntersectionPruner
from synth.filter.dfta_pruner import DFTAPruner
from synth.filter.obs_eq_pruner import ObsEqPruner
from synth.filter.syntactic_pruner import (
    UseAllVariablesPruner,
    FunctionPruner,
    SyntacticPruner,
    SetPruner,
)
from synth.filter.constraints import add_constraints
