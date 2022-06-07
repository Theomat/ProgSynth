from hashlib import new
from operator import add
from textwrap import indent
from typing import Dict, Generator, List, Optional, Tuple,Set

class Bucket:
    def __init__(self,tup: List[int] = [0,0,0]):
        self.elems = []
        self.size = len(tup)
        for elem in tup:
            self.elems.append(elem)
    
    def __str__(self):
        s = "("
        for elem in self.elems:
            s += "{},".format(elem)
        s = s[:-1] + ")"
        return s
    
    def __repr__(self):
        return str(self)

    def __lt__(self,other):
        
        if self.size == 0:
            return False

        if self.elems[0] > other.elems[0]:
            return True
        elif self.elems[0] == other.elems[0]:
            return Bucket(self.elems[1:]).__lt__(Bucket(other.elems[1:]))
        else:
            return False

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return all(self.elems[i] == other.elems[i] for i in range(self.size))

    def add(self, other):
        if(self.size == other.size):
            self.elems = list(map(add, self.elems, other.elems))
        else:
            raise RuntimeError("size mismatch, Bucket{}: {}, Bucket{}: {}".format(self,self.size,other,other.size))

    def add_prob_uniform(self, probability: float):
        index = self.size - int(probability*self.size) - 1
        self.elems[index] += 1
