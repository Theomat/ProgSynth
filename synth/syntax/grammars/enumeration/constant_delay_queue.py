from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(order=True, frozen=True)
class CostTuple:
    cost: float
    combinations: List[List[int]]

    def __repr__(self) -> str:
        return f"({self.cost}, {self.combinations})"


from colorama import Fore as F


class CDQueue:
    """
    Init:
        push*

    Usage:
        pop
        push*
        update

    """

    def __init__(self, maxi: int, k: int) -> None:
        # multiply otherwise when you get exactly maxi then it is equal to 0
        self.maxi = maxi * (k + 1) / k
        self.k = k + 1
        self.clear()

    def update(self) -> None:
        """
        Update its internal representation., should be done after all elements have been pushed.
        """
        if self.nelements > 0:
            while self.cells[self.translation][0] == 0:
                self.translation = (self.translation + 1) % self.k
                self.n += 1
            if self.nelements == 1:
                self.mini = self.cells[self.translation][1].cost
                self.start = self.mini
                self.n = 0
            else:
                self.mini = self.start + self.maxi * self.n / self.k

    def clear(self) -> None:
        """
        Clear this queue of all of its elements.
        """
        self.mini = None
        self.cells = [(0, None) for _ in range(self.k)]
        self.translation = 0
        self.nelements = 0
        self.start = None
        self.n = 0

    def push(self, element: CostTuple) -> None:
        # print("PUSH")
        if self.mini is None:
            self.mini = element.cost
            self.start = self.mini
            self.n = 0
        assert element.cost - self.mini <= self.maxi
        # a = f"[PUSH] {F.LIGHTYELLOW_EX}BEFORE{F.RESET}:{self}"
        # print(f"[PUSH] {F.LIGHTYELLOW_EX}BEFORE{F.RESET}:", self)
        # b = f"\t\t\t{element}"
        # print("\t\t\t", element)
        self.__push__(
            element, element.cost - self.mini, self.maxi, self.cells, self.translation
        )
        # if x or abs(element.cost - 11.040864126152853) <= 1e-3:
        #     print("n=", self.n, "unit:", self.maxi / self.k)
        #     print(a)
        #     print(b)
        #     print(f"[PUSH] {F.GREEN}AFTER{F.RESET}:", self)

    def __push__(
        self,
        element: CostTuple,
        cost: float,
        maxi: float,
        cells: List,
        translation: int,
        add: bool = True,
    ) -> None:
        stack = []
        while True:
            unit = maxi / self.k
            lbi = int(cost / maxi * self.k)
            index = (lbi + translation) % self.k
            # assert cost >= 0, f"cost:{cost} element:{element} queue:{self}"
            nelem, val = cells[index]
            # print("\tfound:", nelem, "val=", val)
            if nelem == 0:
                # print("\t\tfilling...")
                cells[index] = (1, element)
                if add:
                    self.nelements += 1
                    for c, i in stack:
                        c[i][0] += 1
                return

            elif nelem == 1:
                if abs(val.cost - element.cost) > 1:
                    # print(f"\t\tsplitting with {val}... diff={abs(val.cost - element.cost)} ")

                    cells[index] = [2, [(0, None) for _ in range(self.k)]]
                    self.__push__(
                        val,
                        cost + val.cost - element.cost - lbi * unit,
                        unit,
                        cells[index][1],
                        0,
                        add=False,
                    )
                    self.__push__(
                        element,
                        cost - lbi * unit,
                        unit,
                        cells[index][1],
                        0,
                        add=False,
                    )
                    if add:
                        self.nelements += 1
                        for c, i in stack:
                            c[i][0] += 1
                    return
                else:
                    val.combinations.extend(element.combinations)
                    # print("\t\tmerging...")
                    return
            else:
                # print("\t\trecursive...")
                stack.append((cells, index))
                cost = cost - lbi * unit
                maxi = unit
                cells = val
                translation = 0
        return

    def __cleanup__(self, cells: Tuple, index: int) -> None:
        # print(f"[CLEANUP] {F.LIGHTBLUE_EX}BEFORE{F.RESET}:", cells[index])
        n, _ = cells[index]
        if n > 1:
            if n == 2:
                cells[index] = (1, self.__pop__(cells[index]))
                # print("AFTER:", self)
            else:
                cells[index][0] -= 1
        else:
            cells[index] = (0, None)
        # print("[CLEANUP] AFTER:", cells[index])

    def pop(self) -> CostTuple:
        popped = self.__pop__(self.cells[self.translation])
        self.__cleanup__(self.cells, self.translation)
        self.nelements -= 1
        self.last_pop = popped
        return popped

    def __pop__(self, cells) -> Optional[CostTuple]:
        nelems, val = cells
        if nelems <= 1:
            return val
        for i, elem in enumerate(val):
            n, v = elem
            if n > 0:
                if n == 1:
                    val[i] = (0, None)
                    return v
                else:
                    popped = self.__pop__(elem)
                    if popped is not None:
                        self.__cleanup__(val, i)
                        return popped
                    else:
                        val[i] = (0, None)
        return None

    def peek(self) -> CostTuple:
        # term, content = self.cells[self.translation]
        # assert content is not None
        popped = self.__peek__(self.cells[self.translation])
        # assert popped is not None
        return popped

    def __peek__(self, cells: Tuple) -> Optional[CostTuple]:
        nelems, val = cells
        if nelems <= 1:
            return val
        for elem in val:
            n, v = elem
            if v is not None:
                if n == 1:
                    return v
                else:
                    popped = self.__peek__(elem)
                    if popped is not None:
                        return popped
        return None

    def size(self) -> int:
        return sum(self.__size__(el) for el in self.cells)

    def __size__(self, cell) -> int:
        terminal, val = cell
        if val is None:
            return 0
        elif terminal == 1:
            return 1
        else:
            return sum(max(1, self.__size__(elem)) for elem in val)

    def is_empty(self) -> bool:
        return self.nelements == 0

    def __repr__(self) -> str:
        ordered = self.cells[self.translation :] + self.cells[: self.translation]
        out = f"CDQueue[size={self.nelements}/{self.size()}, mini={self.mini}, maxi={self.mini + self.maxi * (self.k / (self.k + 1))}/{self.mini + self.maxi}, k={self.k}]\n\t{ordered}"
        return out

    def __len__(self) -> int:
        return self.nelements
