import argparse
from cgi import test
from enum import Enum
from os import abort
import string
import time
import itertools
import random
from operator import add
from heapq import heappop, heappush
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple,Set

from synth.syntax.bucket import Bucket
from synth.syntax.concrete.heap_search import Enumerator
from synth.syntax.program import Program, Function, Variable
from synth.syntax.concrete.concrete_cfg import NonTerminal
from synth.syntax.concrete.concrete_pcfg import ConcretePCFG

@dataclass(order=True, frozen=True)
class BHeapElement:
    priority: Bucket
    program: Program = field(compare=False)

class BSEnumerator(Enumerator):

    def __init__(self, G: ConcretePCFG) -> None:

        super().__init__(G)
        
        # Initialisation heaps
        ## 0. compute max probability
        self.bucket_tuples = self.G.compute_max_bucket_tuples()

        ## 1. add P(max(S1),max(S2), ...) to self.heaps[S] for all S -> P(S1, S2, ...)
        for S in reversed(self.rules):
            for P in self.rules[S]:

                program = self.G.max_bucket_tuple[(S, P)]
                hash_program = hash(program)

                # Remark: the program cannot already be in self.heaps[S]
                assert hash_program not in self.hash_table_program[S]

                self.hash_table_program[S].add(hash_program)

                # we assume that the programs from max_probability
                # are represented by the same object
                self.hash_table_global[hash_program] = program

                heappush(
                    self.heaps[S],
                    BHeapElement(self.bucket_tuples[program][S], program),
                )

        # 2. call query(S, None) for all non-terminal symbols S, from leaves to root
        for S in reversed(self.rules):
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

        # otherwise the successor is the next element in the heap
        try:
            element = heappop(self.heaps[S])
            succ = element.program
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
                        new_bucket = Bucket()
                        new_bucket.add_prob_uniform(self.G.rules[S][F][1])
                        for arg, S3 in zip(new_arguments, self.G.rules[S][F][0]):
                            new_bucket.add(self.bucket_tuples[arg][S3])
                            
                        heappush(
                            self.heaps[S], 
                            BHeapElement(new_bucket, new_program)
                        )
                        self.bucket_tuples[new_program][S] = new_bucket

        if isinstance(succ, Variable):
            return succ  # if succ is a variable, there is no successor so we stop here

        return succ


def enumerate_pcfg_bucket(G: ConcretePCFG) -> BSEnumerator:  
    return BSEnumerator(G)