from collections import defaultdict
from heapq import heappush, heappop
from typing import (
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
from abc import ABC, abstractmethod

from synth.syntax.grammars.enumeration.heap_search import HeapElement, Bucket
from synth.syntax.program import Program, Function, Variable
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.type_system import Type
from synth.utils.ordered import Ordered

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


def __wrap__(el: Union[U, List[U]]) -> Union[U, Tuple[U, ...]]:
    if isinstance(el, list):
        return tuple(el)
    return el


class UHSEnumerator(ABC, Generic[U, V, W]):
    def __init__(self, G: ProbUGrammar[U, V, W]) -> None:
        self.G = G
        symbols = [S for S in self.G.rules]

        # self.heaps[S] is a heap containing programs generated from the non-terminal S
        self.heaps: Dict[Tuple[Type, U], List[HeapElement]] = {S: [] for S in symbols}

        self._start_heap: List[Tuple[HeapElement, Tuple[Type, U]]] = []

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
            program = self.start_query()
            if program is None:
                break
            yield program

    def __iter__(self) -> Generator[Program, None, None]:
        return self.generator()

    def __init_helper__(
        self,
        Si: Tuple[Type, U],
        information: W,
        i: int,
        args: List[Program],
        nargs: int,
    ) -> Generator[List[Program], None, None]:
        # Try to init sub Tuple[Type, U] in case they were not initialised
        self.__init_non_terminal__(Si)
        args.append(self.max_priority[Si])
        if i + 1 >= nargs:
            yield args
            return
        for information, lst in self.G.derive_all(information, Si, args[-1]):
            Sip1 = lst[-1][0]
            for sol in self.__init_helper__(Sip1, information, i + 1, args[:], nargs):
                yield sol

    def __init_non_terminal__(self, S: Tuple[Type, U]) -> None:
        if S in self._init:
            return
        self._init.add(S)
        # 1) Compute max probablities
        best_program: Optional[Program] = None
        best_priority: Optional[Ordered] = None
        for P in self.G.rules[S]:
            nargs = self.G.arguments_length_for(S, P)
            P_unique: Program = P
            if nargs > 0:
                for information, Si, v in self.G.derive(
                    self.G.start_information(), S, P
                ):
                    for arguments in self.__init_helper__(
                        Si, information, 0, [], nargs
                    ):
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
        for P in self.G.rules[S]:
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

    def start_query(self) -> Optional[Program]:
        if len(self._init) == 0:
            # INIT start heap
            for start in self.G.starts:
                prog = self.query(start, None)
                assert prog
                prio = self.compute_priority(start, prog)
                heappush(self._start_heap, (HeapElement(prio, prog), start))
        if len(self._start_heap) == 0:
            return None
        elem, start = heappop(self._start_heap)
        prog = self.query(start, elem.program)
        if prog is not None:
            prio = self.compute_priority(start, prog)
            heappush(self._start_heap, (HeapElement(prio, prog), start))
        return elem.program

    def __add_successors_to_heap__(
        self, succ: Function, S: Tuple[Type, U], Si: Tuple[Type, U], info: W, i: int
    ) -> None:
        if i >= len(succ.arguments):
            return
        # Si is non-terminal symbol used to derive the i-th argument
        succ_sub_program = self.query(Si, succ.arguments[i])
        if succ_sub_program:
            new_arguments = succ.arguments[:]
            new_arguments[i] = succ_sub_program
            new_program = self.__return_unique__(Function(succ.function, new_arguments))
            hash_new_program = hash(new_program)
            if hash_new_program not in self.hash_table_program[S]:
                self.hash_table_program[S].add(hash_new_program)
                # try:
                priority: Ordered = self.compute_priority(S, new_program)
                heappush(self.heaps[S], HeapElement(priority, new_program))
                # except KeyError:
                # pass
        for info, lst in self.G.derive_all(info, Si, succ.arguments[i]):
            Sip1 = lst[-1][0]
            self.__add_successors_to_heap__(succ, S, Sip1, info, i + 1)

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
            for info, lst in self.G.derive_all(self.G.start_information(), S, F):
                Si = lst[-1][0]
                self.__add_successors_to_heap__(succ, S, Si, info, 0)

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

    def __prob__(
        self, succ: Function, S: Tuple[Type, U], Si: Tuple[Type, U], info: W, i: int
    ) -> float:
        # Si is non-terminal symbol used to derive the i-th argument
        arg = succ.arguments[i]
        probability = self.probabilities[arg][Si]
        if i + 1 >= len(succ.arguments):
            return probability
        for info, lst in self.G.derive_all(info, Si, arg):
            Sip1 = lst[-1][0]
            prob = self.__prob__(succ, S, Sip1, info, i + 1)
            if prob >= 0:
                return prob * probability
        return -1

    def compute_priority(self, S: Tuple[Type, U], new_program: Program) -> float:
        # Try to hit cached probability
        if new_program in self.probabilities and S in self.probabilities[new_program]:
            return -self.probabilities[new_program][S]
        if isinstance(new_program, Function):
            F = new_program.function
            for information, lst in self.G.derive_all(self.G.start_information(), S, F):
                Si = lst[-1][0]
                v = __wrap__(lst[-1][-1])
                # We guarantee that F is a Primitive
                probability = self.G.probabilities[S][F][v]  # type: ignore
                prob = self.__prob__(new_program, S, Si, information, 0)
                if prob >= 0:
                    break
            probability = prob * probability
        else:
            possibles = self.G.derive_all(self.G.start_information(), S, new_program)
            assert len(possibles) == 1
            v = __wrap__(possibles[0][-1][-1][-1])
            probability = self.G.probabilities[S][new_program][v]  # type: ignore
        self.probabilities[new_program][S] = probability
        return -probability


def enumerate_prob_u_grammar(G: ProbUGrammar[U, V, W]) -> UHeapSearch[U, V, W]:
    return UHeapSearch(G)


class BucketSearch(UHSEnumerator[U, V, W]):
    def __init__(self, G: ProbUGrammar[U, V, W], bucket_size: int) -> None:
        super().__init__(G)
        self.bucket_tuples: Dict[Program, Dict[Tuple[Type, U], Bucket]] = defaultdict(
            lambda: {}
        )
        self.bucket_size = bucket_size

    def __prob__(
        self, succ: Function, S: Tuple[Type, U], Si: Tuple[Type, U], info: W, i: int
    ) -> Optional[Bucket]:
        # Si is non-terminal symbol used to derive the i-th argument
        arg = succ.arguments[i]
        bucket = self.bucket_tuples[arg][Si]
        if i + 1 >= len(succ.arguments):
            return bucket
        for info, lst in self.G.derive_all(info, Si, arg):
            Sip1 = lst[-1][0]
            prob = self.__prob__(succ, S, Sip1, info, i + 1)
            if prob:
                return prob + bucket
        return None

    def compute_priority(self, S: Tuple[Type, U], new_program: Program) -> Bucket:
        new_bucket = Bucket(self.bucket_size)
        if isinstance(new_program, Function):
            F = new_program.function
            for information, lst in self.G.derive_all(self.G.start_information(), S, F):
                Si = lst[-1][0]
                v = __wrap__(lst[-1][-1])
                # We guarantee that F is a Primitive
                probability = self.G.probabilities[S][F][v]  # type: ignore
                prob = self.__prob__(new_program, S, Si, information, 0)
                if prob:
                    break
            new_bucket.add_prob_uniform(probability)
            assert prob is not None
            new_bucket += prob
        else:
            possibles = self.G.derive_all(self.G.start_information(), S, new_program)
            assert len(possibles) == 1
            v = __wrap__(possibles[0][-1][-1][-1])
            probability = self.G.probabilities[S][new_program][v]  # type: ignore
            new_bucket.add_prob_uniform(probability)
        self.bucket_tuples[new_program][S] = new_bucket
        return new_bucket


def enumerate_bucket_prob_u_grammar(
    G: ProbUGrammar[U, V, W], bucket_size: int
) -> BucketSearch[U, V, W]:
    return BucketSearch(G, bucket_size)
