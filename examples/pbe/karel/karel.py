from typing import List, Optional
from dataclasses import dataclass, field

from synth.syntax import (
    DSL,
    auto_type,
)
from synth.semantic import DSLEvaluator

import numpy as np

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class KarelProg:
    def then(self, s2: "KarelProg") -> "KarelProg":
        return KarelThen(self, s2)


@dataclass(frozen=True)
class KarelRepeat(KarelProg):
    subroutine: KarelProg
    n: int


@dataclass(frozen=True)
class KarelThen(KarelProg):
    s1: KarelProg
    s2: KarelProg


@dataclass(frozen=True)
class KarelAction(KarelProg):
    action: str


@dataclass(frozen=True)
class KarelCond(KarelProg):
    cond: str
    negated: bool = field(default=False)

    def neg(self) -> "KarelCond":
        return KarelCond(self.cond, not self.negated)


@dataclass(frozen=True)
class KarelWhile(KarelProg):
    subroutine: KarelProg
    cond: KarelCond


@dataclass(frozen=True)
class KarelITE(KarelProg):
    cond: KarelCond
    yes: KarelProg
    no: Optional[KarelProg]


class KarelWorld:

    DIRECTION_TOP = 0
    DIRECTION_LEFT = 1
    DIRECTION_BOTTOM = 2
    DIRECTION_RIGHT = 3

    def __init__(self, width: int, height: int) -> None:
        self.grid = np.zeros((width, height))
        self.markers = np.zeros_like(self.grid)
        self.reset()

    def reset(self) -> None:
        self.karel = (0, 0)
        self.current_markers = self.markers.copy()
        self.direction = self.DIRECTION_RIGHT

    def isFrontClear(self) -> bool:
        x, y = self.karel
        width, height = self.grid.shape
        new = self.karel
        if self.direction == self.DIRECTION_BOTTOM:
            new = (x, y + 1)
        elif self.direction == self.DIRECTION_LEFT:
            new = (x - 1, y)
        elif self.direction == self.DIRECTION_TOP:
            new = (x, y - 1)
        else:
            new = (x + 1, y)
        if min(new) < 0 or new[0] >= width or new[1] >= height:
            return False
        return self.grid[new] <= 0

    def act(self, command: str) -> None:
        if command == "move":
            if not self.isFrontClear():
                return
            x, y = self.karel
            if self.direction == self.DIRECTION_BOTTOM:
                self.karel = (x, y + 1)
            elif self.direction == self.DIRECTION_LEFT:
                self.karel = (x - 1, y)
            elif self.direction == self.DIRECTION_TOP:
                self.karel = (x, y - 1)
            else:
                self.karel = (x + 1, y)
        elif command == "turnLeft":
            self.direction -= 1
            if self.direction < 0:
                self.direction = 3
        elif command == "turnRight":
            self.direction += 1
            if self.direction > 3:
                self.direction = 0
        elif command == "putMarker":
            self.current_markers[self.karel] = 1
        elif command == "pickMarker":
            self.current_markers[self.karel] = 0
        else:
            raise Exception(f"invalid command:{command}")

    def eval(self, cond: str) -> bool:
        if cond == "frontIsClear":
            return self.isFrontClear()
        elif cond == "leftIsClear":
            self.act("turnLeft")
            isClear = self.isFrontClear()
            self.act("turnRight")
            return isClear
        elif cond == "rightIsClear":
            self.act("turnRight")
            isClear = self.isFrontClear()
            self.act("turnLeft")
            return isClear
        elif cond == "markersPresent":
            return self.markers[self.karel]
        elif cond == "noMarkersPresent":
            return not self.markers[self.karel]
        raise Exception(f"invalid cond:{cond}")

    def exec(self, prog: KarelProg) -> bool:
        if isinstance(prog, KarelAction):
            self.act(prog.action)
        elif isinstance(prog, KarelCond):
            out = self.eval(prog.cond)
            if prog.negated:
                out = not out
            return out
        elif isinstance(prog, KarelThen):
            self.exec(prog.s1)
            self.exec(prog.s2)
        elif isinstance(prog, KarelRepeat):
            for _ in range(prog.n):
                self.exec(prog.subroutine)
        elif isinstance(prog, KarelWhile):
            n = 0
            max_it = self.grid.shape[0] * self.grid.shape[1]
            while self.exec(prog.cond) and n < max_it:
                self.exec(prog.subroutine)
                n += 1
        elif isinstance(prog, KarelITE):
            if self.exec(prog.cond):
                self.exec(prog.yes)
            elif prog.no is not None:
                self.exec(prog.no)
        return False

    def state(self) -> tuple:
        out = self.markers * 2 + self.grid + self.current_markers * 4
        out[self.karel] += 8
        return tuple(tuple(x) for x in out)

    def show(self) -> None:
        plt.figure()

        # Draw Karel
        x, y = self.karel
        xs = [x, x + 1 / 2, x + 1]
        ys = [y, y + 1, y]
        if self.direction == self.DIRECTION_RIGHT:
            xs = [x, x + 1, x]
            ys = [y, y + 1 / 2, y + 1]
        elif self.direction == self.DIRECTION_LEFT:
            xs = [x + 1, x, x + 1]
            ys = [y, y + 1 / 2, y + 1]
        elif self.direction == self.DIRECTION_TOP:
            xs = [x, x + 1 / 2, x + 1]
            ys = [y + 1, y, y + 1]
        plt.fill(xs, ys, "blue")
        # Draw Grid
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):

                if self.grid[x, y] > 0:
                    plt.fill([x, x, x + 1, x + 1], [y + 1, y, y, y + 1], "g")
                elif self.current_markers[x, y] > 0:
                    plt.scatter([x + 1 / 2], [y + 1 / 2], color="r", marker="D", s=12)
        plt.xlim(0, self.grid.shape[0])
        plt.ylim(0, self.grid.shape[1])
        plt.xticks(list(range(self.grid.shape[0] + 1)))
        plt.yticks(list(range(self.grid.shape[1] + 1)))
        plt.grid()
        plt.show()


__syntax = auto_type(
    {
        "run": "world -> stmt -> result",
        "then": "stmt -> stmt -> stmt",
        "move": "stmt",
        "turnRight": "stmt",
        "turnLeft": "stmt",
        "pickMarker": "stmt",
        "putMarker": "stmt",
        "frontIsClear": "cond",
        "leftIsClear": "cond",
        "rightIsClear": "cond",
        "markersPresent": "cond",
        "noMarkersPresent": "cond",
        "not": "cond -> cond",
        "if": "cond -> stmt -> stmt",
        "repeat": "int -> stmt -> stmt",
        "while": "cond -> stmt -> stmt",
    }
)


def __run__(world: KarelWorld, prog: KarelProg) -> tuple:
    world.reset()
    world.exec(prog)
    # print(prog)
    # world.show()
    return world.state()


__semantics = {
    "run": lambda grid: lambda s: __run__(grid, s),
    "then": lambda s1: lambda s2: s1.then(s2),
    "move": KarelAction("move"),
    "turnRight": KarelAction("turnRight"),
    "turnLeft": KarelAction("turnLeft"),
    "pickMarker": KarelAction("pickMarker"),
    "putMarker": KarelAction("putMarker"),
    "frontIsClear": KarelCond("frontIsClear"),
    "leftIsClear": KarelCond("leftIsClear"),
    "rightIsClear": KarelCond("rightIsClear"),
    "markersPresent": KarelCond("markersPresent"),
    "noMarkersPresent": KarelCond("noMarkersPresent"),
    "not": lambda c: c.neg(),
    "if": lambda c: lambda s: KarelITE(c, s, None),
    "repeat": lambda n: lambda s: KarelRepeat(s, n),
    "while": lambda c: lambda s: KarelWhile(s, c),
}
# Add constants
for i in range(3, 10):
    __syntax[str(i)] = auto_type("int")
    __semantics[str(i)] = i

__forbidden_patterns = {
    ("not", 0): {"not", "markersPresent"},
}

dsl = DSL(__syntax, __forbidden_patterns)
evaluator = DSLEvaluator(dsl.instantiate_semantics(__semantics))
lexicon = []


constraints = [
    "then ^then _",
    "while _ ^while,repeat",
    "repeat _ ^while,repeat",
]


def pretty_print_inputs(inputs: List[KarelWorld]) -> str:
    world = inputs[0]
    world.show()
    return "shown"
