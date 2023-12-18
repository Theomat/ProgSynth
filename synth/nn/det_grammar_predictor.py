from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Set,
    Tuple,
    Optional,
    TypeVar,
    Union,
)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from synth.syntax.grammars.tagged_det_grammar import TaggedDetGrammar, ProbDetGrammar
from synth.syntax.grammars.det_grammar import DerivableProgram, DetGrammar
from synth.syntax.program import Constant, Primitive, Program, Variable
from synth.syntax.type_system import Type

A = TypeVar("A")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


class TensorLogProbDetGrammar(TaggedDetGrammar[Tensor, U, V, W]):
    """
    Special version to compute with Tensors
    """

    def log_probability(
        self,
        program: Program,
        start: Optional[Tuple[Type, U]] = None,
    ) -> Tensor:
        device = self.tags[self.start][list(self.tags[self.start].keys())[0]].device
        return self.reduce_derivations(
            lambda current, S, P, _: current + self.tags[S][P],
            torch.zeros((1,)).to(device),
            program,
            start,
        )

    def to_prob_det_grammar(self) -> ProbDetGrammar[U, V, W]:
        probabilities = {
            S: {P: np.exp(w.item()) for P, w in self.tags[S].items()}
            for S in self.rules
        }
        return ProbDetGrammar(self.grammar, probabilities)


class DetGrammarPredictorLayer(nn.Module, Generic[A, U, V, W]):
    """

    Parameters:
    ------------
    - input_size: int - the input size of the tensor to this layer
    - grammars: Iterable[DetGrammar[U, V, W]] - the set of all supported grammars
    - variable_probability: float = 0.2 - the probability mass of all variable at any given derivation level
    """

    def __init__(
        self,
        input_size: int,
        grammars: Iterable[DetGrammar[U, V, W]],
        abstraction: Callable[[Tuple[Type, U]], A],
        variable_probability: float = 0.2,
    ):
        super(DetGrammarPredictorLayer, self).__init__()

        self.grammar_dictionary = {
            grammar.type_request: grammar for grammar in grammars
        }
        self.variable_probability = variable_probability

        # Compute all pairs (A, P) where A is an abstraction of S
        self.abs2real: Dict[A, Set[Tuple[Type, U]]] = defaultdict(set)
        self.real2abs: Dict[Tuple[Type, U], A] = {}

        self.all_pairs: Dict[Optional[A], Set[Primitive]] = {}
        for grammar in grammars:
            for S in grammar.rules:
                abstract = abstraction(S)
                self.abs2real[abstract].add(S)
                self.real2abs[S] = abstract

                key = abstract
                if not key in self.all_pairs:
                    self.all_pairs[key] = set()
                for P in grammar.rules[S]:
                    if not isinstance(P, (Variable, Constant)):
                        self.all_pairs[key].add(P)

        output_size = sum(len(self.all_pairs[S]) for S in self.all_pairs)
        self.output_size = output_size
        self.abs2index: Dict[
            Optional[A],
            Tuple[int, int, Dict[Primitive, int]],
        ] = {}
        current_index = 0
        for okey, set_for_key in self.all_pairs.items():
            self.abs2index[okey] = (
                current_index,
                len(set_for_key),
                {P: i for i, P in enumerate(self.all_pairs[okey])},
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

        returns: (batch_size, self.output_size)
        """
        y: Tensor = self.log_probs_predictor(x)
        return y

    def tensor2log_prob_grammar(
        self,
        x: Tensor,
        type_request: Type,
        total_variable_order: bool = True,
    ) -> TensorLogProbDetGrammar[U, V, W]:
        """

        Parameters:
        ------------
        - x: Tensor - the tensor to be transformed into a TensorLogProbDetGrammar
        - type_request: Type - the type request of the PCFG
        - total_variable_order: bool = True - reduce very slighlty (1e-7) some variable probabilities to ensure they are totally ordered in terms of probablities

        """
        device = x.device
        self.__normalize__(x, x)
        grammar = self.grammar_dictionary[type_request]
        tags: Dict[Tuple[Type, U], Dict[DerivableProgram, Tensor]] = {}
        for S in grammar.rules:
            tags[S] = {}
            key = self.real2abs[S]
            start, length, symbol2index = self.abs2index[key]
            y = x[start : start + length]

            # List of all variables derivable from S
            variables: List[Variable] = []
            constants: List[Constant] = []
            # For each derivation parse probabilities
            for P in grammar.rules[S]:
                cpy_P = P
                if isinstance(P, Primitive):
                    primitive_index = symbol2index[P]
                    tags[S][cpy_P] = y[primitive_index]
                elif isinstance(P, Variable):
                    V: Variable = P  # ensure typing
                    variables.append(V)
                    # All variables together have probability mass self.variable_probability
                    # then the probability of selecting a variable is uniform
                elif isinstance(P, Constant):
                    C: Constant = P  # ensure typing
                    constants.append(C)
                else:
                    continue
            # If there are variables we need to normalise
            total = sum(np.exp(tags[S][P].item()) for P in tags[S])
            if variables or constants:
                var_probability = self.variable_probability
                if total > 0:
                    # Normalise rest
                    to_add: float = np.log((1 - self.variable_probability) / total)
                    for O in tags[S]:
                        tags[S][O] = tags[S][O] + to_add
                else:
                    # There are no other choices than variables
                    var_probability = 1
                # Normalise variable probability
                normalised_variable_logprob: float = np.log(
                    var_probability / (len(variables) + len(constants))
                )
                for P in variables:
                    tags[S][P] = torch.tensor(normalised_variable_logprob).to(device)
                    # Trick to allow a total ordering on variables
                    if total_variable_order:
                        normalised_variable_logprob = np.log(
                            np.exp(normalised_variable_logprob) - 1e-7
                        )
                for P in constants:
                    tags[S][P] = torch.tensor(normalised_variable_logprob).to(device)
            elif total > 0:
                # We still need to normalise probabilities
                # Since all derivations aren't possible
                to_add = np.log(1 / total)
                for O in tags[S]:
                    tags[S][O] = tags[S][O] + to_add
        grammar = TensorLogProbDetGrammar(grammar, tags)
        return grammar

    def encode(
        self,
        program: Program,
        type_request: Type,
        device: Union[torch.device, str, Literal[None]] = None,
    ) -> Tensor:
        out: Tensor = torch.zeros((self.output_size), device=device)
        grammar = self.grammar_dictionary[type_request]
        grammar.reduce_derivations(__reduce_encoder__, (self, out), program)
        return out

    def __normalize__(self, src: Tensor, dst: Tensor) -> None:
        # Normalize
        if len(dst.shape) == 1:
            for _, (start, length, _) in self.abs2index.items():
                dst[start : start + length] = F.log_softmax(
                    src[start : start + length], dim=-1
                )

        else:
            for _, (start, length, _) in self.abs2index.items():
                dst[:, start : start + length] = F.log_softmax(
                    src[:, start : start + length], dim=-1
                )

    def loss_mse(
        self,
        programs: Iterable[Program],
        type_requests: Iterable[Type],
        batch_outputs: Tensor,
        reduce: Optional[Callable[[Tensor], Tensor]] = torch.mean,
    ) -> Tensor:
        target = torch.log(
            1e-5
            + torch.stack(
                [
                    self.encode(prog, tr, device=batch_outputs.device)
                    for prog, tr in zip(programs, type_requests)
                ]
            )
        ).to(device=batch_outputs.device)
        dst = torch.zeros_like(batch_outputs)
        self.__normalize__(batch_outputs, dst)
        out = F.mse_loss(dst, target)
        if reduce:
            out = reduce(out)
        return out

    def loss_negative_log_prob(
        self,
        programs: Iterable[Program],
        log_pgrammars: Iterable[TensorLogProbDetGrammar[U, V, W]],
        reduce: Optional[Callable[[Tensor], Tensor]] = torch.mean,
        length_normed: bool = True,
    ) -> Tensor:
        """
        Computes the negative log prob of each solution program.
        This works independently of the abstraction used.
        """
        if length_normed:
            log_prob_list = [
                log_pgrammar.log_probability(p) / p.size()
                for p, log_pgrammar in zip(programs, log_pgrammars)
            ]
        else:
            log_prob_list = [
                log_pgrammar.log_probability(p)
                for p, log_pgrammar in zip(programs, log_pgrammars)
            ]
        out = -torch.stack(log_prob_list)
        if reduce:
            out = reduce(out)
        return out


def __reduce_encoder__(
    t: Tuple[DetGrammarPredictorLayer[A, U, V, W], Tensor],
    S: Tuple[Type, U],
    P: DerivableProgram,
    _: V,
) -> Tuple[DetGrammarPredictorLayer[A, U, V, W], Tensor]:
    if isinstance(P, Primitive):
        G, tensor = t
        start, __, symbol2index = G.abs2index[G.real2abs[S]]
        tensor[start + symbol2index[P]] = 1
    return t
