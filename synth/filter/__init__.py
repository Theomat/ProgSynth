"""
Module that contains anything relevant to pruning
"""
from synth.filter.filter import Filter, UnionFilter, IntersectionFilter
from synth.filter.dfta_filter import DFTAFilter
from synth.filter.obs_eq_filter import ObsEqFilter
from synth.filter.local_stateless_filter import LocalStatelessFilter
from synth.filter.syntactic_filter import (
    UseAllVariablesFilter,
    FunctionFilter,
    SyntacticFilter,
    SetFilter,
)
from synth.filter.constraints import add_constraints, add_dfta_constraints
