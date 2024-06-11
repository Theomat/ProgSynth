
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(order=True, frozen=True)
class CostTuple:
    cost: float
    combinations: List[List[int]]

    def __repr__(self) -> str:
        return f"({self.cost}, {self.combinations})"
    

class CDQueue:
    """
    Init:
        push*

    Usage:
        pop
        push*
        update
    
    """

    def __init__(self, maxi: float, k: int, precision: float) -> None:
        # multiply otherwise when you get exactly maxi then it is equal to 0
        self.maxi = maxi * (1 + precision)
        self.k = k
        self.precision = precision
        self.clear()

    def update(self):
        """
        Update its internal representation., should be done after all elements have been pushed.
        """
        # print("\t\t\ttranslation old:", self.translation)
        # print("\t\t:", self)
        if self.nelements > 0:
            while self.__peek__(self.cells[self.translation]) is None:
                self.translation = (self.translation + 1) % self.k
                self.n += 1
            self.mini = self.start + self.maxi * self.n / self.k # type: ignore
        # print("\t\t\ttranslation new:", self.translation)
        # print("\t\t[UPDATE]:", self)

    def clear(self):
        """
        Clear this queue of all of its elements.
        """
        self.mini = None
        self.cells = [None for _ in range(self.k)]
        self.translation = 0
        self.nelements = 0
        self.start = None
        self.n = 0


    def push(self, element: CostTuple):
        if self.mini is None:
            self.mini = element.cost
            self.start = element.cost
        self.__push__(element, element.cost - self.mini, self.maxi, self.cells, self.translation)


    def __push__(self, element: CostTuple, cost: float, maxi: float, cells: List, translation: int):
        while True:
            unit = maxi / self.k
            lbi = int(cost / maxi * self.k)
            index = (lbi + translation) % self.k
            # print("\tindex=", index, "for cost=", cost)
            assert self.translation == 0 or cost >= 0, f"cost:{cost} element:{element} queue:{self}"
            val = cells[index]
            if val is None:
                # print("\t\tfilling...")
                cells[index] = element
                self.nelements += 1
                return

            elif isinstance(val, CostTuple):
                if abs(val.cost - element.cost) > self.precision:
                    # print("\t\tsplitting...")

                    cells[index] = [None for _ in range(self.k)]
                    self.__push__(element, cost - lbi * unit, unit, cells[index], 0)
                    self.push(val)
                    self.nelements -= 1
                    return
                else:
                    val.combinations.extend(element.combinations)
                    # print("\t\tmerging...")
                    return
            else:
                # print("\t\trecursive...")
                cost = cost - lbi * unit
                maxi = unit
                cells = val
                translation = 0

    def pop(self) -> CostTuple:
        i = 0 
        while i < self.k:
            ci = (self.translation + i) % self.k
            popped, should_pop = self.__pop__(self.cells[ci])
            # print("ci=", ci, "popped=", popped, "should_pop=", should_pop, "cells=", self.cells[ci])
            if popped is not None:
                if should_pop:
                    self.cells[ci] = None
                # print("\t\tcells:", self.cells)
                self.nelements -= 1
                return popped
            i += 1
            self.cells[ci] = None

    def __pop__(self, cells) -> Tuple[Optional[CostTuple], bool]:
        if not isinstance(cells, List):
            return cells, True
        for i, elem in enumerate(cells):
            if elem is not None:
                if isinstance(elem, CostTuple):
                    cells[i] = None
                    return elem, i == self.k - 1
                else:
                    popped, should_pop = self.__pop__(elem)
                    if popped is not None:
                        if should_pop:
                            cells[i] = None
                        return popped, False
                    else:
                        cells[i] = None
        return None, False
    
    def peek(self) -> CostTuple:
        i = 0 
        while i < self.k:
            ci = (self.translation + i) % self.k
            popped = self.__peek__(self.cells[ci])
            if popped is not None:
                # print("at i=", i , "ci=", ci, "returned:", popped, "type:", type(popped))
                return popped
            i += 1
            self.cells[ci] = None

    def __peek__(self, cells) -> Optional[CostTuple]:
        if not isinstance(cells, List):
            return cells
        for elem in cells:
            if elem is not None:
                if isinstance(elem, CostTuple):
                    return elem
                else:
                    popped = self.__peek__(elem)
                    if popped is not None:
                        return popped
        return None
                
    def size(self) -> int:
        return self.__size__(self.cells)

    def __size__(self, cell) -> int:
        if cell is None:
            return 0
        elif isinstance(cell, CostTuple):
            return 1
        else:
            return sum(max(1, self.__size__(elem)) for elem in cell)
        
    def is_empty(self) -> bool:
        return self.nelements == 0

    def __repr__(self) -> str:
        out = f"CDQueue[size={self.nelements}/{self.size()}, mini={self.mini}, maxi={self.mini + self.maxi}, k={self.k}, precision={self.precision}]\n\t{self.cells[self.translation:] + self.cells[:self.translation]}"
        return out
    
    def __len__(self) -> int:
        return self.nelements
