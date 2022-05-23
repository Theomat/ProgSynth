from heapq import heappush, heappop
from typing import Dict, Generator, List, Optional, Set
from dataclasses import dataclass, field

from synth.syntax.program import Program, Function, Variable
from synth.syntax.concrete.concrete_cfg import NonTerminal
from synth.syntax.concrete.concrete_pcfg import ConcretePCFG


@dataclass(order=True, frozen=True)
class HeapElement:
    priority: float
    program: Program = field(compare=False)


class HSEnumerator:
    hash_table_global: Dict[int, Program] = {}

    def __init__(self, G: ConcretePCFG) -> None:
        self.current: Optional[Program] = None

        self.G = G
        self.start = G.start
        self.rules = G.rules
        self.symbols = [S for S in self.rules]

        # self.heaps[S] is a heap containing programs generated from the non-terminal S
        self.heaps: Dict[NonTerminal, List[HeapElement]] = {S: [] for S in self.symbols}

        # the same program can be pushed in different heaps, with different probabilities
        # however, the same program cannot be pushed twice in the same heap

        # self.succ[S][P] is the successor of P from S
        self.succ: Dict[NonTerminal, Dict[int, Program]] = {S: {} for S in self.symbols}

        # self.hash_table_program[S] is the set of hashes of programs
        # ever added to the heap for S
        self.hash_table_program: Dict[NonTerminal, Set[int]] = {
            S: set() for S in self.symbols
        }

        # self.hash_table_global[hash] = P maps
        # hashes to programs for all programs ever added to some heap
        self.hash_table_global = {}

        # Initialisation heaps

        ## 0. compute max probability
        self.probabilities = self.G.compute_max_probability()

        ## 1. add P(max(S1),max(S2), ...) to self.heaps[S] for all S -> P(S1, S2, ...)
        for S in reversed(self.rules):
            for P in self.rules[S]:
                program = self.G.max_probability[(S, P)]
                hash_program = hash(program)

                # Remark: the program cannot already be in self.heaps[S]
                assert hash_program not in self.hash_table_program[S]

                self.hash_table_program[S].add(hash_program)

                # we assume that the programs from max_probability
                # are represented by the same object
                self.hash_table_global[hash_program] = program

                # print("adding to the heap", program, program.probability[S])
                heappush(
                    self.heaps[S],
                    HeapElement(-self.probabilities[program][S], program),
                )

        # 2. call query(S, None) for all non-terminal symbols S, from leaves to root
        for S in reversed(self.rules):
            self.query(S, None)

    def __return_unique__(self, P: Program) -> Program:
        """
        ensures that if a program appears in several heaps,
        it is represented by the same object,
        so we do not evaluate it several times
        """
        hash_P = hash(P)
        if hash_P in self.hash_table_global:
            return self.hash_table_global[hash_P]
        else:
            self.hash_table_global[hash_P] = P
            return P

    def generator(self) -> Generator[Program, None, None]:
        """
        A generator which outputs the next most probable program
        """
        while True:
            program = self.query(self.start, self.current)
            if program is None:
                break
            self.current = program
            yield program

    def __iter__(self) -> Generator[Program, None, None]:
        return self.generator()

    def query(self, S: NonTerminal, program: Optional[Program]) -> Optional[Program]:
        """
        computing the successor of program from S
        """
        if program:
            hash_program = hash(program)
        else:
            hash_program = 123891

        # if we have already computed the successor of program from S, we return its stored value
        if hash_program in self.succ[S]:
            # print("already computed the successor of S, it's ", S, program, self.succ[S][hash_program])
            return self.succ[S][hash_program]

        # otherwise the successor is the next element in the heap
        try:
            element = heappop(self.heaps[S])
            succ = element.program
            # print("found succ in the heap", S, program, succ)
        except:
            return None  # the heap is empty: there are no successors from S

        self.succ[S][hash_program] = succ  # we store the succesor

        # now we need to add all potential successors of succ in heaps[S]
        if isinstance(succ, Function):
            F = succ.function

            for i in range(len(succ.arguments)):
                # non-terminal symbol used to derive the i-th argument
                S2 = self.G.rules[S][F][0][i]
                succ_sub_program = self.query(S2, succ.arguments[i])
                if succ_sub_program:
                    new_arguments = succ.arguments[:]
                    new_arguments[i] = succ_sub_program

                    new_program: Program = Function(F, new_arguments)
                    new_program = self.__return_unique__(new_program)
                    hash_new_program = hash(new_program)

                    if hash_new_program not in self.hash_table_program[S]:
                        self.hash_table_program[S].add(hash_new_program)
                        probability = self.G.rules[S][F][1]
                        for arg, S3 in zip(new_arguments, self.G.rules[S][F][0]):
                            probability *= self.probabilities[arg][S3]
                        heappush(self.heaps[S], HeapElement(-probability, new_program))
                        self.probabilities[new_program][S] = probability

        if isinstance(succ, Variable):
            return succ  # if succ is a variable, there is no successor so we stop here

        return succ


def enumerate_pcfg(G: ConcretePCFG) -> HSEnumerator:
    return HSEnumerator(G)
