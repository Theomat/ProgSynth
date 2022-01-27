from typing import Callable, Dict, Iterable, List, Tuple, Optional, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from synth.syntax.concrete.concrete_cfg import ConcreteCFG, Context
from synth.syntax.concrete.concrete_pcfg import ConcretePCFG
from synth.syntax.dsl import DSL
from synth.syntax.program import Function, Primitive, Program, Variable
from synth.syntax.type_system import Arrow, Type


LogPRules = Dict[Context, Dict[Program, Tuple[List[Context], Tensor]]]


class ConcreteLogPCFG:
    """
    Special version of ConcretePCFG to compute with Tensors
    """

    def __init__(
        self, start: Context, rules: LogPRules, max_program_depth: int, type_req: Type
    ):
        self.start = start
        self.rules = rules
        self.max_program_depth = max_program_depth
        self.hash_table_programs: Dict[int, Program] = {}

        self.hash = hash(str(rules))
        self.type_request = type_req

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, ConcreteLogPCFG)
            and self.type_request == o.type_request
            and self.rules == o.rules
        )

    def __str__(self) -> str:
        s = "Print a LogPCFG\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += "#\n {}\n".format(S)
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                s += "   {} - {}: {}     {}\n".format(P, P.type, args_P, w.item())
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def log_probability(self, P: Program, S: Optional[Context] = None) -> Tensor:
        """
        Compute the log probability of a program P generated from the non-terminal S
        """
        S = S or self.start
        if isinstance(P, Function):
            F = P.function
            args_P = P.arguments
            probability = self.rules[S][F][1]

            for i, arg in enumerate(args_P):
                probability += self.log_probability(arg, self.rules[S][F][0][i])
            return probability

        elif isinstance(P, (Variable, Primitive)):
            return self.rules[S][P][1]

        print("log_probability_program", P)
        assert False

    def to_pcfg(self) -> ConcretePCFG:
        rules = {
            S: {P: (args, np.exp(w.item())) for P, (args, w) in self.rules[S].items()}
            for S in self.rules
        }
        return ConcretePCFG(self.start, rules, self.max_program_depth, True)


class BigramsPredictorLayer(nn.Module):
    """

    Parameters:
    ------------
    - input_size: int - the input size of the tensor to this layer
    - dsl: DSL - the dsl with which this predictor is used
    - cfgs: Iterable[ConcreteCFG] - the set of all supported CFG
    - variable_probability: float = 0.2 - the probability mass of all variable at any given derivation level
    """

    def __init__(
        self,
        input_size: int,
        dsl: DSL,
        cfgs: Iterable[ConcreteCFG],
        variable_probability: float = 0.2,
    ):
        super(BigramsPredictorLayer, self).__init__()

        self.cfg_dictionary = {cfg.type_request: cfg for cfg in cfgs}
        self.variable_probability = variable_probability
        self.dsl = dsl

        self.symbol2index = {
            symbol: index for index, symbol in enumerate(self.dsl.list_primitives)
        }
        func_primitives: List[Union[Primitive, Variable]] = [
            p for p in self.dsl.list_primitives if isinstance(p.type, Arrow)
        ]
        variables_used_as_arguments = set(
            S.predecessors[0][0]
            for cfg in cfgs
            for S in cfg.rules.keys()
            if len(S.predecessors) > 0 and isinstance(S.predecessors[0][0], Variable)
        )
        func_primitives += list(variables_used_as_arguments)

        self.parent2index = {
            symbol: index for index, symbol in enumerate(func_primitives)
        }

        # IMPORTANT: we do not predict variables!
        self.number_of_primitives = len(self.symbol2index)
        self.number_of_parents = len(self.parent2index) + 1  # could be None
        self.maximum_arguments = max(
            len(p.type.arguments()) if isinstance(p.type, Arrow) else 0
            for p in self.dsl.list_primitives
        )
        self.log_probs_predictor = nn.Linear(
            input_size,
            self.number_of_parents * self.maximum_arguments * self.number_of_primitives,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        batch_IOs is a tensor of size
        (batch_size, input_size)

        returns: (batch_size, self.number_of_parents, self.maximum_arguments, self.number_of_primitives)
        """
        y: Tensor = self.log_probs_predictor(x)
        z = F.log_softmax(
            y.view(
                (
                    -1,
                    self.number_of_parents,
                    self.maximum_arguments,
                    self.number_of_primitives,
                )
            ),
            dim=-1,
        )
        return z

    def tensor2pcfg(
        self,
        x: Tensor,
        type_request: Type,
        total_variable_order: bool = True,
        device: str = "cpu",
    ) -> ConcreteLogPCFG:
        """

        Parameters:
        ------------
        - x: Tensor - the tensor to be transformed into a PCFG
        - type_request: Type - the type request of the PCFG
        - total_variable_order: bool = True - reduce very slighlty (1e-7) some variable probabilities to ensure they are totally ordered in terms of probablities

        """
        cfg = self.cfg_dictionary[type_request]
        rules: LogPRules = {}
        for S in cfg.rules:
            rules[S] = {}
            # Compute parent_index and argument_number
            if S.predecessors:
                parent_index = self.parent2index[S.predecessors[0][0]]
                argument_number = S.predecessors[0][1]
            else:
                parent_index = len(self.parent2index)  # no parent => None
                argument_number = 0
            # List of all variables derivable from S
            variables: List[Variable] = []
            # For each derivation parse probabilities
            for P in cfg.rules[S]:
                cpy_P = P
                if isinstance(P, Primitive):
                    primitive_index = self.symbol2index[P]
                    rules[S][cpy_P] = (
                        cfg.rules[S][P],
                        x[parent_index, argument_number, primitive_index],
                    )
                else:
                    V: Variable = P  # ensure typing
                    variables.append(V)
                    # All variables together have probability mass self.variable_probability
                    # then the probability of selecting a variable is uniform
            # If there are variables we need to normalise
            total = sum(np.exp(rules[S][P][1].item()) for P in rules[S])
            if variables:
                var_probability = self.variable_probability
                if total > 0:
                    # Normalise rest
                    to_add: float = np.log((1 - self.variable_probability) / total)
                    for O in rules[S]:
                        rules[S][O] = rules[S][O][0], rules[S][O][1] + to_add
                else:
                    # There are no other choices than variables
                    var_probability = 1
                # Normalise variable probability
                normalised_variable_logprob: float = np.log(
                    var_probability / len(variables)
                )
                for P in variables:
                    rules[S][P] = cfg.rules[S][P], torch.tensor(
                        normalised_variable_logprob
                    ).to(device)
                    # Trick to allow a total ordering on variables
                    if total_variable_order:
                        normalised_variable_logprob = np.log(
                            np.exp(normalised_variable_logprob) - 1e-7
                        )
            else:
                # We still need to normalise probabilities
                # Since all derivations aren't possible
                total = sum(np.exp(rules[S][P][1].item()) for P in rules[S])
                to_add = np.log(1 / total)
                for O in rules[S]:
                    rules[S][O] = rules[S][O][0], rules[S][O][1] + to_add
        grammar = ConcreteLogPCFG(cfg.start, rules, cfg.max_program_depth, type_request)
        return grammar


def loss_negative_log_prob(
    programs: Iterable[Program],
    log_pcfgs: Iterable[ConcreteLogPCFG],
    reduce: Optional[Callable[[Tensor], Tensor]] = torch.mean,
) -> Tensor:
    log_prob_list = [
        log_pcfg.log_probability(p) for p, log_pcfg in zip(programs, log_pcfgs)
    ]
    out = -torch.stack(log_prob_list)
    if reduce:
        out = reduce(out)
    return out
