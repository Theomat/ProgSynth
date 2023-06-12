from synth.syntax import (
    DSL,
    auto_type,
)
from synth.semantic import DSLEvaluator

from karel_runtime import (
    KarelAction,
    KarelCond,
    KarelITE,
    KarelWhile,
    KarelRepeat,
    KarelProg,
    KarelWorld,
)

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
        "ifelse": "cond -> stmt -> stmt -> stmt",
        "repeat": "int -> stmt -> stmt",
        "while": "cond -> stmt -> stmt",
    }
)


def __run__(world: KarelWorld, prog: KarelProg) -> tuple:
    world.reset()
    world.exec(prog)
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
    "ifelse": lambda c: lambda s1: lambda s2: KarelITE(c, s1, s2),
    "repeat": lambda n: lambda s: KarelRepeat(s, n),
    "while": lambda c: lambda s: KarelWhile(s, c),
}
# Add constants
for i in range(20):
    __syntax[str(i)] = auto_type("int")
    __semantics[str(i)] = i


dsl = DSL(__syntax)
evaluator = DSLEvaluator(__semantics)
lexicon = []
