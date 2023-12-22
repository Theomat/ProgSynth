from typing import List
from synth.syntax import (
    DSL,
    auto_type,
)
from synth.semantic import DSLEvaluator

import numpy as np

import matplotlib.pyplot as plt


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

    def act(self, command: str) -> "KarelWorld":
        if command == "move":
            if not self.isFrontClear():
                return self
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
        return self

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
        "then": "(world -> world) -> (world -> world) -> (world -> world)",
        "move": "world -> world",
        "turnRight": "world -> world",
        "turnLeft": "world -> world",
        "pickMarker": "world -> world",
        "putMarker": "world -> world",
        "frontIsClear": "world -> bool",
        "leftIsClear": "world -> bool",
        "rightIsClear": "world -> bool",
        "markersPresent": "world -> bool",
        "noMarkersPresent": "world -> bool",
        "not": "'a [bool | (world -> bool)] -> 'a [bool | (world -> bool)]",
        "ite": "bool -> (world -> world) -> (world -> world) -> (world -> world)",
        "repeat": "int -> (world -> world) -> (world -> world)",
        "while": "(world -> bool) -> (world -> world) -> (world -> world)",
    }
)


def __while(w: KarelWorld, c, s) -> KarelWorld:
    n = 0
    while c(w) and n < 10000:
        w = s(w)
        n += 1
    return w


__semantics = {
    "then": lambda f: lambda g: lambda z: f(g(z)),
    "move": lambda w: w.act("move"),
    "turnRight": lambda w: w.act("turnRight"),
    "turnLeft": lambda w: w.act("turnLeft"),
    "pickMarker": lambda w: w.act("pickMarker"),
    "putMarker": lambda w: w.act("putMarker"),
    "frontIsClear": lambda w: w.eval("frontIsClear"),
    "leftIsClear": lambda w: w.eval("leftIsClear"),
    "rightIsClear": lambda w: w.eval("rightIsClear"),
    "markersPresent": lambda w: w.eval("markersPresent"),
    "noMarkersPresent": lambda w: w.eval("noMarkersPresent"),
    "not": lambda c: not c if isinstance(c, bool) else lambda w: not c(w),
    "if": lambda c: lambda ifblock: lambda elseblock: ifblock if c else elseblock,
    "repeat": lambda n: lambda s: lambda w: [s(w) for _ in range(n)][-1],
    "while": lambda c: lambda s: lambda w: __while(w, c, s),
}
# Add constants
for i in range(3, 10):
    __syntax[str(i)] = auto_type("int")
    __semantics[str(i)] = i

__forbidden_patterns = {("not", 0): {"not", "markersPresent"}, ("then", 0): {"then"}}

dsl = DSL(__syntax, __forbidden_patterns)
dsl.instantiate_polymorphic_types(3)
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
