from collections import defaultdict
from heapq import heappush, heappop
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from bisect import bisect_left

from synth.syntax.grammars.grammar import DerivableProgram
from synth.syntax.program import Program, Function, Variable
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.type_system import Type
from synth.utils.ordered import Ordered

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


@dataclass(order=True, frozen=True)
class HeapElement:
    priority: Ordered
    program: Program = field(compare=False)

    def __repr__(self) -> str:
        return f"({self.priority}, {self.program})"


class UHSEnumerator(ABC, Generic[U, V, W]):
    def __init__(self, G: ProbUGrammar[U, V, W]) -> None:
        self.current: Optional[Program] = None

        self.G = G
        self.starts = G.starts
        self.rules = G.rules
        symbols = [S for S in self.rules]

        # self.heaps[S] is a heap containing programs generated from the non-terminal S
        self.heaps: Dict[Tuple[Type, U], List[HeapElement]] = {S: [] for S in symbols}

        # the same program can be pushed in different heaps, with different probabilities
        # however, the same program cannot be pushed twice in the same heap

        # self.succ[S][P] is the successor of P from S
        self.succ: Dict[Tuple[Type, U], Dict[int, Program]] = {S: {} for S in symbols}

        # self.hash_table_program[S] is the set of hashes of programs
        # ever added to the heap for S
        self.hash_table_program: Dict[Tuple[Type, U], Set[int]] = {
            S: set() for S in symbols
        }

        # self.hash_table_global[hash] = P maps
        # hashes to programs for all programs ever added to some heap
        self.hash_table_global: Dict[int, Program] = {}

        self._init: Set[Tuple[Type, U]] = set()

        self.max_priority: Dict[
            Union[Tuple[Type, U], Tuple[Tuple[Type, U], Program]], Program
        ] = {}

        self._starts: Dict[Tuple[Type, U], Optional[Program]] = {
            start: None for start in self.G.starts
        }
        self._start_orders: List[Tuple[HeapElement, Tuple[Type, U]]] = []

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
            program = self.__init_query__()
            if program is None:
                break
            self.current = program
            yield program

    def __iter__(self) -> Generator[Program, None, None]:
        return self.generator()

    def __init_non_terminal__(self, S: Tuple[Type, U]) -> None:
        if S in self._init:
            return
        self._init.add(S)
        # 1) Compute max probablities
        best_program: Optional[Program] = None
        best_priority: Optional[Ordered] = None
        for P in self.rules[S]:
            nargs = self.G.arguments_length_for(S, P)
            P_unique: Program = P
            if nargs > 0:
                possibles = self.G.derive(self.G.start_information(), S, P)
                cases: List[Tuple[W, Tuple[Type, U], List[Program]]] = [
                    (information, current, []) for information, current, _ in possibles
                ]
                while cases:
                    information, current, arguments = cases.pop()
                    if len(arguments) < nargs:
                        self.__init_non_terminal__(current)
                        # Try to init sub Tuple[Type, U] in case they were not initialised
                        arguments.append(self.max_priority[current])
                        next_possibles = self.G.derive_all(
                            information, current, arguments[-1]
                        )
                        for info, lst in next_possibles:
                            current = lst[-1][0]
                            cases.append((info, current, arguments[:]))
                    else:
                        new_program = Function(
                            function=P_unique,
                            arguments=arguments,
                        )
                        P_unique = new_program
                        priority = self.compute_priority(S, P_unique)
                        self.max_priority[(S, P)] = P_unique
                        if not best_priority or priority < best_priority:
                            best_program = P_unique
                            best_priority = priority
            else:
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
        self.query(S, None)

    def __add_successor__(
        self,
        S: Tuple[Type, U],
        F: DerivableProgram,
        S2: Tuple[Type, U],
        information: W,
        succ: Function,
        i: int = 0,
    ) -> None:
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
        next_possibles = self.G.derive_all(information, S2, succ.arguments[i])
        for information, lst in next_possibles:
            S2 = lst[-1][0]
            self.__add_successor__(S, F, S2, information, succ, i + 1)

    def __init_query__(self) -> Optional[Program]:
        if len(self._start_orders) == 0:
            for start in self.G.starts:
                succ = self.query(start, None)
                if succ is not None:
                    addition = (
                        HeapElement(self.compute_priority(start, succ), succ),
                        start,
                    )
                    self._start_orders.append(addition)
            self._start_orders.sort()
        elem, start = self._start_orders.pop(0)
        program = elem.program
        succ = self.query(start, program)
        if succ is not None:
            addition = (HeapElement(self.compute_priority(start, succ), succ), start)
            i = bisect_left(self._start_orders, addition)
            self._start_orders.insert(i, addition)
        return program

    def query(self, S: Tuple[Type, U], program: Optional[Program]) -> Optional[Program]:
        """
        computing the successor of program from S
        """
        if S not in self._init:
            self.__init_non_terminal__(S)
        if program:
            hash_program = hash(program)
        else:
            hash_program = 123891

        # if we have already computed the successor of program from S, we return its stored value
        if hash_program in self.succ[S]:
            return self.succ[S][hash_program]

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
            possibles = self.G.derive_all(self.G.start_information(), S, F)
            for information, lst in possibles:
                S2 = lst[-1][0]
                self.__add_successor__(S, F, S2, information, succ, 0)  # type: ignore

        if isinstance(succ, Variable):
            return succ  # if succ is a variable, there is no successor so we stop here

        return succ

    @abstractmethod
    def compute_priority(self, S: Tuple[Type, U], new_program: Program) -> Ordered:
        pass


class UHeapSearch(UHSEnumerator[U, V, W]):
    def __init__(self, G: ProbUGrammar[U, V, W]) -> None:
        super().__init__(G)
        self.probabilities: Dict[Program, Dict[Tuple[Type, U], float]] = defaultdict(
            lambda: {}
        )

    def __next_prob__(
        self, args: List[Program], S: Tuple[Type, U], information: W
    ) -> List[float]:
        if len(args) == 0:
            return [1.0]
        arg = args[0]
        probability = self.probabilities[arg].get(S, 0)
        if probability == 0:
            return []
        out = []
        for information, lst in self.G.derive_all(information, S, arg):
            S2 = lst[-1][0]
            p = self.__next_prob__(args[1:], S2, information)
            if len(p) == 0:
                continue
            for prob in p:
                out.append(prob * probability)
        return out

    def compute_priority(self, S: Tuple[Type, U], new_program: Program) -> float:
        if new_program in self.probabilities and S in self.probabilities[new_program]:
            return -self.probabilities[new_program][S]
        if isinstance(new_program, Function):
            F = new_program.function
            # We guarantee that F is a Primitive
            possibles = self.G.derive_all(self.G.start_information(), S, F)
            out = []
            for information, lst in possibles:
                probability = self.G.probabilities[S][F][lst[-1][1]]  # type: ignore
                S2 = lst[-1][0]
                probs = self.__next_prob__(new_program.arguments, S2, information)
                if len(probs) == 0:
                    continue
                out += [x * probability for x in probs]
            assert len(out) == 1
            return out[0]

        else:
            lst = self.G.derive_all(self.G.start_information(), S, new_program)[0][1]
            probability = self.G.probabilities[S][new_program][lst[-1][1]]  # type: ignore
        self.probabilities[new_program][S] = probability
        return -probability


def enumerate_prob_u_grammar(G: ProbUGrammar[U, V, W]) -> UHeapSearch[U, V, W]:
    return UHeapSearch(G)


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
            if self.elems[i] < other.elems[i]:
                return True
            elif self.elems[i] > other.elems[i]:
                return False
        return False

    def __gt__(self, other: "Bucket") -> bool:
        return other.__lt__(self)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Bucket) and all(
            self.elems[i] == other.elems[i] for i in range(self.size)
        )

    def __le__(self, other: "Bucket") -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other: "Bucket") -> bool:
        return self.__gt__(other) or self.__eq__(other)

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

    def __add__(self, other: "Bucket") -> "Bucket":
        if self.size == other.size:
            bucket = Bucket(self.size)
            bucket.elems = [self.elems[i] + other.elems[i] for i in range(self.size)]
            return bucket
        else:
            raise RuntimeError(
                "size mismatch, Bucket{}: {}, Bucket{}: {}".format(
                    self, self.size, other, other.size
                )
            )

    def add_prob_uniform(self, probability: float) -> "Bucket":
        """
        Given a probability add 1 in the relevant bucket assuming buckets are linearly distributed.
        """
        index = self.size - int(probability * self.size) - 1
        self.elems[index] += 1
        return self


class UBucketSearch(UHSEnumerator[U, V, W]):
    def __init__(self, G: ProbUGrammar[U, V, W], bucket_size: int) -> None:
        super().__init__(G)
        self.bucket_tuples: Dict[Program, Dict[Tuple[Type, U], Bucket]] = defaultdict(
            lambda: {}
        )
        self.bucket_size = bucket_size

    def __next_prob__(
        self, args: List[Program], S: Tuple[Type, U], information: W
    ) -> List[Bucket]:
        if len(args) == 0:
            return [Bucket(self.bucket_size)]
        arg = args[0]
        probability = self.bucket_tuples[arg].get(S, None)
        if probability is None:
            return []
        out = []
        for information, lst in self.G.derive_all(information, S, arg):
            S2 = lst[-1][0]
            p = self.__next_prob__(args[1:], S2, information)
            if len(p) == 0:
                continue
            for prob in p:
                out.append(prob + probability)
        return out

    def compute_priority(self, S: Tuple[Type, U], new_program: Program) -> Bucket:
        if new_program in self.bucket_tuples and S in self.bucket_tuples[new_program]:
            return self.bucket_tuples[new_program][S]
        new_bucket = Bucket(self.bucket_size)
        if isinstance(new_program, Function):
            F = new_program.function
            # We guarantee that F is a Primitive
            possibles = self.G.derive_all(self.G.start_information(), S, F)
            out = []
            for information, lst in possibles:
                probability = self.G.probabilities[S][F][lst[-1][1]]  # type: ignore

                S2 = lst[-1][0]
                probs = self.__next_prob__(new_program.arguments, S2, information)
                if len(probs) == 0:
                    continue
                new_bucket.add_prob_uniform(probability)
                out += probs
            assert len(out) == 1
            return out[0]
        else:
            lst = self.G.derive_all(self.G.start_information(), S, new_program)[0][1]
            probability = self.G.probabilities[S][new_program][lst[-1][1]]  # type: ignore
            new_bucket.add_prob_uniform(probability)
        self.bucket_tuples[new_program][S] = new_bucket
        return new_bucket


def enumerate_bucket_prob_u_grammar(
    G: ProbUGrammar[U, V, W], bucket_size: int
) -> UBucketSearch[U, V, W]:
    return UBucketSearch(G, bucket_size)
