from typing import Any, Callable, Dict, Iterable, List, Set, Tuple, Optional, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from synth.nn.utils import one_hot_encode_primitives
from synth.syntax.concrete.concrete_cfg import ConcreteCFG, NonTerminal
from synth.syntax.concrete.concrete_pcfg import ConcretePCFG
from synth.syntax.dsl import DSL
from synth.syntax.program import Constant, Function, Primitive, Program, Variable
from synth.syntax.type_system import Type


LogPRules = Dict[NonTerminal, Dict[Program, Tuple[List[NonTerminal], Tensor]]]


class ConcreteLogPCFG:
    """
    Special version of ConcretePCFG to compute with Tensors
    """

    def __init__(
        self,
        start: NonTerminal,
        rules: LogPRules,
        max_program_depth: int,
        type_req: Type,
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

    def log_probability(self, P: Program, S: Optional[NonTerminal] = None) -> Tensor:
        """
        Compute the log probability of a program P generated from the non-terminal S
        """
        S = S or self.start
        if isinstance(P, Function):
            F = P.function
            args_P = P.arguments
            probability = self.rules[S][F][1]

            for i, arg in enumerate(args_P):
                probability = probability + self.log_probability(
                    arg, self.rules[S][F][0][i]
                )
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
    - cfgs: Iterable[ConcreteCFG] - the set of all supported CFG
    - variable_probability: float = 0.2 - the probability mass of all variable at any given derivation level
    """

    def __init__(
        self,
        input_size: int,
        cfgs: Iterable[ConcreteCFG],
        variable_probability: float = 0.2,
    ):
        super(BigramsPredictorLayer, self).__init__()

        self.cfg_dictionary = {cfg.type_request: cfg for cfg in cfgs}
        self.variable_probability = variable_probability

        # Compute all pairs (S, P) where S has lost depth information
        self.all_pairs: Dict[
            Optional[Tuple[Union[Primitive, Variable], int]], Set[Primitive]
        ] = {}
        for cfg in cfgs:
            for S in cfg.rules:
                key = S.predecessors[0] if S.predecessors else None
                if not key in self.all_pairs:
                    self.all_pairs[key] = set()
                for P in cfg.rules[S]:
                    if not isinstance(P, (Variable, Constant)):
                        self.all_pairs[key].add(P)

        output_size = sum(len(self.all_pairs[S]) for S in self.all_pairs)

        self.s2index: Dict[
            Optional[Tuple[Union[Primitive, Variable], int]],
            Tuple[int, int, Dict[Primitive, int]],
        ] = {}
        current_index = 0
        for key, set_for_key in self.all_pairs.items():
            self.s2index[key] = (
                current_index,
                len(set_for_key),
                {P: i for i, P in enumerate(self.all_pairs[key])},
            )
            current_index += len(set_for_key)

        self.log_probs_predictor = nn.Linear(
            input_size,
            output_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        batch_IOs is a tensor of size
        (batch_size, input_size)

        returns: (batch_size, self.number_of_parents, self.maximum_arguments, self.number_of_primitives)
        """
        y: Tensor = self.log_probs_predictor(x)
        z = torch.ones_like(y)
        for _, (start, length, _) in self.s2index.items():
            z[:, start : start + length] = F.log_softmax(
                y[:, start : start + length], dim=-1
            )

        return z

    def tensor2pcfg(
        self,
        x: Tensor,
        type_request: Type,
        total_variable_order: bool = True,
    ) -> ConcreteLogPCFG:
        """

        Parameters:
        ------------
        - x: Tensor - the tensor to be transformed into a PCFG
        - type_request: Type - the type request of the PCFG
        - total_variable_order: bool = True - reduce very slighlty (1e-7) some variable probabilities to ensure they are totally ordered in terms of probablities

        """
        device = x.device
        cfg = self.cfg_dictionary[type_request]
        rules: LogPRules = {}
        for S in cfg.rules:
            rules[S] = {}
            key = S.predecessors[0] if S.predecessors else None
            start, length, symbol2index = self.s2index[key]
            y = x[start : start + length]

            # List of all variables derivable from S
            variables: List[Variable] = []
            # For each derivation parse probabilities
            for P in cfg.rules[S]:
                cpy_P = P
                if isinstance(P, Primitive):
                    primitive_index = symbol2index[P]
                    rules[S][cpy_P] = (
                        cfg.rules[S][P],
                        y[primitive_index],
                    )
                elif isinstance(P, Variable):
                    V: Variable = P  # ensure typing
                    variables.append(V)
                    # All variables together have probability mass self.variable_probability
                    # then the probability of selecting a variable is uniform
                else:
                    continue
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
                to_add = np.log(1 / total)
                for O in rules[S]:
                    rules[S][O] = rules[S][O][0], rules[S][O][1] + to_add
        grammar = ConcreteLogPCFG(cfg.start, rules, cfg.max_program_depth, type_request)
        return grammar


class PrimitivePredictorLayer(nn.Module):
    """

    Parameters:
    ------------
    - input_size: int - the input size of the tensor to this layer
    - dsl: DSL - the DSL to be supported
    - variable_probability: float = 0.2 - the probability mass of all variable at any given derivation level
    """

    def __init__(
        self,
        input_size: int,
        dsl: DSL,
        variable_probability: float = 0.2,
    ):
        super(PrimitivePredictorLayer, self).__init__()
        self.dsl = dsl
        self.variable_probability = variable_probability
        self.primitives = dsl.list_primitives[:]
        self.P2index = {p: i for i, p in enumerate(self.primitives)}
        output_size = len(self.primitives)
        self.log_probs_predictor = nn.Linear(
            input_size,
            output_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        batch_IOs is a tensor of size
        (batch_size, input_size)

        returns: (batch_size, len(self.primitives) + 1) in logits
        """
        y: Tensor = self.log_probs_predictor(x)
        z =  F.softmax(y)
        return z

    def tensor2pcfg(
        self,
        x: Tensor,
        type_request: Type,
        total_variable_order: bool = True,
        **kwargs: Any
    ) -> ConcreteLogPCFG:
        """

        Parameters:
        ------------
        - x: Tensor - the tensor to be transformed into a PCFG
        - type_request: Type - the type request of the PCFG
        - total_variable_order: bool = True - reduce very slighlty (1e-7) some variable probabilities to ensure they are totally ordered in terms of probablities
        -- **kwargs are passed to ConcreteCFG.from_dsl

        """
        device = x.device
        cfg = ConcreteCFG.from_dsl(self.dsl, type_request, **kwargs)
        rules: LogPRules = {}
        x = torch.log(x)
        for S in cfg.rules:
            rules[S] = {}

            # List of all variables derivable from S
            variables: List[Variable] = []
            # For each derivation parse probabilities
            for P in cfg.rules[S]:
                cpy_P = P
                if isinstance(P, Primitive):
                    primitive_index = self.P2index[P]
                    rules[S][cpy_P] = (
                        cfg.rules[S][P],
                        x[primitive_index],
                    )
                elif isinstance(P, Variable):
                    V: Variable = P  # ensure typing
                    variables.append(V)
                    # All variables together have probability mass self.variable_probability
                    # then the probability of selecting a variable is uniform
                else:
                    continue
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
                to_add = np.log(1 / total)
                for O in rules[S]:
                    rules[S][O] = rules[S][O][0], rules[S][O][1] + to_add
        grammar = ConcreteLogPCFG(cfg.start, rules, cfg.max_program_depth, type_request)
        return grammar

    def loss(
        self,
        programs: Iterable[Program],
        batch: Tensor,
        loss: Callable[[Tensor, Tensor], Tensor] = F.cross_entropy,
        reduce: Callable[[Tensor], Tensor] = torch.mean,
    ) -> Tensor:
        encoded_programs = torch.stack(
            [
                one_hot_encode_primitives(p, self.P2index, len(self.primitives))
                for p in programs
            ]
        ).to(batch.device)
        out = loss(encoded_programs, batch)
        if reduce:
            out = reduce(out)
        return out


def loss_negative_log_prob(
    programs: Iterable[Program],
    log_pcfgs: Iterable[ConcreteLogPCFG],
    reduce: Optional[Callable[[Tensor], Tensor]] = torch.mean,
    length_normed: bool = True,
) -> Tensor:
    if length_normed:
        log_prob_list = [
            log_pcfg.log_probability(p) / p.length()
            for p, log_pcfg in zip(programs, log_pcfgs)
        ]
    else:
        log_prob_list = [
            log_pcfg.log_probability(p) for p, log_pcfg in zip(programs, log_pcfgs)
        ]
    out = -torch.stack(log_prob_list)
    if reduce:
        out = reduce(out)
    return out
