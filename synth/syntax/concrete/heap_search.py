from collections import defaultdict
from heapq import heappush, heappop
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from synth.syntax.program import Program, Function, Variable
from synth.syntax.concrete.concrete_cfg import NonTerminal
from synth.syntax.concrete.concrete_pcfg import ConcretePCFG
from synth.utils.ordered import Ordered


@dataclass(order=True, frozen=True)
class HeapElement:
    priority: Ordered
    program: Program = field(compare=False)

    def __repr__(self) -> str:
        return f"({self.priority}, {self.program})"


class HSEnumerator(ABC):
    def __init__(self, G: ConcretePCFG) -> None:
        self.current: Optional[Program] = None

        self.G = G
        self.start = G.start
        self.rules = G.rules
        symbols = [S for S in self.rules]

        # self.heaps[S] is a heap containing programs generated from the non-terminal S
        self.heaps: Dict[NonTerminal, List[HeapElement]] = {S: [] for S in symbols}

        # the same program can be pushed in different heaps, with different probabilities
        # however, the same program cannot be pushed twice in the same heap

        # self.succ[S][P] is the successor of P from S
        self.succ: Dict[NonTerminal, Dict[int, Program]] = {S: {} for S in symbols}

        # self.hash_table_program[S] is the set of hashes of programs
        # ever added to the heap for S
        self.hash_table_program: Dict[NonTerminal, Set[int]] = {
            S: set() for S in symbols
        }

        # self.hash_table_global[hash] = P maps
        # hashes to programs for all programs ever added to some heap
        self.hash_table_global: Dict[int, Program] = {}

        self._init: Set[NonTerminal] = set()

        self.max_priority: Dict[
            Union[NonTerminal, Tuple[NonTerminal, Program]], Program
        ] = {}

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

    def __init_non_terminal__(self, S: NonTerminal) -> None:
        # 1) Compute max probablities
        best_program = None
        best_priority: Optional[Ordered] = None
        for P in self.rules[S]:
            args_P, _ = self.rules[S][P]
            P_unique = self.G.return_unique(P)
            if len(args_P) > 0:
                arguments = []
                for arg in args_P:
                    # Try to init sub NonTerminal in case they were not initialised
                    if arg not in self._init:
                        self.__init_non_terminal__(arg)
                    arguments.append(self.max_priority[arg])

                new_program = Function(
                    function=P_unique,
                    arguments=arguments,
                )
                P_unique = self.G.return_unique(new_program)
            priority = self.compute_priority(S, P_unique)
            self.max_priority[(S, P)] = P_unique
            if not best_priority or priority < best_priority:
                best_program = P_unique
                best_priority = priority
        assert best_program
        self.max_priority[S] = best_program

        # 2) add P(max(S1),max(S2), ...) to self.heaps[S]
        for P in self.rules[S]:
            program = self.max_priority[(S, P)]
            hash_program = hash(program)
            # Remark: the program cannot already be in self.heaps[S]
            assert hash_program not in self.hash_table_program[S]
            self.hash_table_program[S].add(hash_program)
            # we assume that the programs from max_probability
            # are represented by the same object
            self.hash_table_global[hash_program] = program
            priority = self.compute_priority(S, program)
            heappush(
                self.heaps[S],
                HeapElement(priority, program),
            )

        # 3) Do the 1st query
        self._init.add(S)
        self.query(S, None)

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
            return self.succ[S][hash_program]

        if S not in self._init:
            self.__init_non_terminal__(S)
        # otherwise the successor is the next element in the heap
        try:
            element = heappop(self.heaps[S])
            succ = element.program
        except:
            return None  # the heap is empty: there are no successors from S

        self.succ[S][hash_program] = succ  # we store the successor

        # now we need to add all potential successors of succ in heaps[S]
        if isinstance(succ, Function):
            F = succ.function

            for i, S2 in enumerate(self.G.rules[S][F][0]):
                # S2 is non-terminal symbol used to derive the i-th argument
                succ_sub_program = self.query(S2, succ.arguments[i])
                if succ_sub_program:
                    new_arguments = succ.arguments[:]
                    new_arguments[i] = succ_sub_program

                    new_program = self.__return_unique__(Function(F, new_arguments))
                    hash_new_program = hash(new_program)

                    if hash_new_program not in self.hash_table_program[S]:
                        self.hash_table_program[S].add(hash_new_program)

                        priority: Ordered = self.compute_priority(S, new_program)
                        heappush(self.heaps[S], HeapElement(priority, new_program))

        if isinstance(succ, Variable):
            return succ  # if succ is a variable, there is no successor so we stop here

        return succ

    @abstractmethod
    def compute_priority(self, S: NonTerminal, new_program: Program) -> Ordered:
        pass


class HeapSearch(HSEnumerator):
    def __init__(self, G: ConcretePCFG) -> None:
        super().__init__(G)
        self.probabilities: Dict[Program, Dict[NonTerminal, float]] = defaultdict(
            lambda: {}
        )

        for S in reversed(self.G.rules):
            self.__init_non_terminal__(S)

    def compute_priority(self, S: NonTerminal, new_program: Program) -> float:
        if isinstance(new_program, Function):
            F = new_program.function
            new_arguments = new_program.arguments
            probability = self.G.rules[S][F][1]
            for arg, S3 in zip(new_arguments, self.G.rules[S][F][0]):
                probability *= self.probabilities[arg][S3]
        else:
            probability = self.G.rules[S][new_program][1]
        self.probabilities[new_program][S] = probability
        return -probability


def enumerate_pcfg(G: ConcretePCFG) -> HeapSearch:
    return HeapSearch(G)


class Bucket(Ordered):
    def __init__(self, size: int = 3):
        self.elems = [0 for _ in range(size)]
        self.size = size

    def __str__(self) -> str:
        s = "("
        for elem in self.elems:
            s += "{},".format(elem)
        s = s[:-1] + ")"
        return s

    def __repr__(self) -> str:
        return str(self)

    def __lt__(self, other: "Bucket") -> bool:
        if self.size == 0:
            return False
        for i in range(self.size):
            if self.elems[i] > other.elems[i]:
                return True
            elif self.elems[i] < other.elems[i]:
                return False
        return False

    def __gt__(self, other: "Bucket") -> bool:
        return other.__lt__(self)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Bucket) and all(
            self.elems[i] == other.elems[i] for i in range(self.size)
        )

    def __iadd__(self, other: "Bucket") -> "Bucket":
        if self.size == other.size:
            for i in range(self.size):
                self.elems[i] += other.elems[i]
            return self
        else:
            raise RuntimeError(
                "size mismatch, Bucket{}: {}, Bucket{}: {}".format(
                    self, self.size, other, other.size
                )
            )

    def add_prob_uniform(self, probability: float) -> None:
        """
        Given a probability add 1 in the relevant bucket assuming buckets are linearly distributed.
        """
        index = self.size - int(probability * self.size) - 1
        self.elems[index] += 1


class BucketSearch(HSEnumerator):
    def __init__(self, G: ConcretePCFG, bucket_size: int) -> None:
        super().__init__(G)
        self.bucket_tuples: Dict[Program, Dict[NonTerminal, Bucket]] = defaultdict(
            lambda: {}
        )
        self.bucket_size = bucket_size

    def compute_priority(self, S: NonTerminal, new_program: Program) -> Bucket:
        new_bucket = Bucket(self.bucket_size)
        if isinstance(new_program, Function):
            F = new_program.function
            new_arguments = new_program.arguments
            new_bucket.add_prob_uniform(self.G.rules[S][F][1])
            for arg, S3 in zip(new_arguments, self.G.rules[S][F][0]):
                new_bucket += self.bucket_tuples[arg][S3]
        else:
            probability = self.G.rules[S][new_program][1]
            new_bucket.add_prob_uniform(probability)
        self.bucket_tuples[new_program][S] = new_bucket
        return new_bucket


def enumerate_bucket_pcfg(G: ConcretePCFG, bucket_size: int) -> BucketSearch:
    return BucketSearch(G, bucket_size)
