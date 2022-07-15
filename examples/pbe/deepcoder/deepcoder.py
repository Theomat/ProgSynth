from typing import Tuple
from synth.semantic import DSLEvaluator
from synth.semantic.evaluator import auto_complete_semantics
from synth.syntax import DSL, INT, Arrow, PolymorphicType, List
from synth.tools.type_constraints import produce_new_syntax_for_constraints

t0 = PolymorphicType("t0")
t1 = PolymorphicType("t1")


def __access__(i, l):
    if i is None:
        return None
    elif (i >= 0 and len(l) > i) or (i < 0 and len(l) >= -i):
        return l[i]
    else:
        return None


def __scanl__(op):
    def aux(l):
        if len(l) == 0:
            return []
        else:
            y = [l[0]]
            for x in l[1:]:
                last = y[-1]
                y.append(op(last, x))
        return y

    return aux


__semantics = {
    "HEAD": lambda l: l[0] if len(l) > 0 else None,
    "TAIL": lambda l: l[-1] if len(l) > 0 else None,
    "ACCESS": lambda i: lambda l: __access__(i, l),
    "MINIMUM": lambda l: min(l) if len(l) > 0 else None,
    "MAXIMUM": lambda l: max(l) if len(l) > 0 else None,
    "LENGTH": lambda l: len(l),
    "COUNT[<0]": lambda l: len([x for x in l if x < 0]),
    "COUNT[>0]": lambda l: len([x for x in l if x > 0]),
    "COUNT[EVEN]": lambda l: len([x for x in l if x % 2 == 0]),
    "COUNT[ODD]": lambda l: len([x for x in l if x % 2 == 1]),
    "SUM": lambda l: sum(l),
    "TAKE": lambda i: lambda l: l[:i],
    "DROP": lambda i: lambda l: l[i:],
    "SORT": lambda l: sorted(l),
    "REVERSE": lambda l: l[::-1],
    "FILTER[<0]": lambda l: [x for x in l if x < 0],
    "FILTER[>0]": lambda l: [x for x in l if x > 0],
    "FILTER[EVEN]": lambda l: [x for x in l if x % 2 == 0],
    "FILTER[ODD]": lambda l: [x for x in l if x % 2 == 1],
    "MAP[+1]": lambda l: [x + 1 for x in l],
    "MAP[-1]": lambda l: [x - 1 for x in l],
    "MAP[*2]": lambda l: [x * 2 for x in l],
    "MAP[/2]": lambda l: [int(x / 2) for x in l],
    "MAP[*3]": lambda l: [x * 3 for x in l],
    "MAP[/3]": lambda l: [int(x / 3) for x in l],
    "MAP[*4]": lambda l: [x * 4 for x in l],
    "MAP[/4]": lambda l: [int(x / 4) for x in l],
    "MAP[**2]": lambda l: [x**2 for x in l],
    "MAP[*-1]": lambda l: [-x for x in l],
    "ZIPWITH[+]": lambda l1: lambda l2: [x + y for (x, y) in zip(l1, l2)],
    "ZIPWITH[-]": lambda l1: lambda l2: [x - y for (x, y) in zip(l1, l2)],
    "ZIPWITH[*]": lambda l1: lambda l2: [x * y for (x, y) in zip(l1, l2)],
    "ZIPWITH[max]": lambda l1: lambda l2: [
        (x if x > y else y) for (x, y) in zip(l1, l2)
    ],
    "ZIPWITH[min]": lambda l1: lambda l2: [
        (y if x > y else x) for (x, y) in zip(l1, l2)
    ],
    "SCAN1L[+]": __scanl__(lambda x, y: x + y),
    "SCAN1L[-]": __scanl__(lambda x, y: x - y),
    "SCAN1L[*]": __scanl__(lambda x, y: x * y),
    "SCAN1L[min]": __scanl__(lambda x, y: min(x, y)),
    "SCAN1L[max]": __scanl__(lambda x, y: max(x, y)),
    # 'MAP': lambda f: lambda l: list(map(f, l)),
}

__primitive_types = {
    "HEAD": Arrow(List(INT), INT),
    "TAIL": Arrow(List(INT), INT),
    "ACCESS": Arrow(INT, Arrow(List(INT), INT)),
    "MINIMUM": Arrow(List(INT), INT),
    "MAXIMUM": Arrow(List(INT), INT),
    "LENGTH": Arrow(List(INT), INT),
    "COUNT[<0]": Arrow(List(INT), INT),
    "COUNT[>0]": Arrow(List(INT), INT),
    "COUNT[EVEN]": Arrow(List(INT), INT),
    "COUNT[ODD]": Arrow(List(INT), INT),
    "SUM": Arrow(List(INT), INT),
    "TAKE": Arrow(INT, Arrow(List(INT), List(INT))),
    "DROP": Arrow(INT, Arrow(List(INT), List(INT))),
    "SORT": Arrow(List(INT), List(INT)),
    "REVERSE": Arrow(List(INT), List(INT)),
    "FILTER[<0]": Arrow(List(INT), List(INT)),
    "FILTER[>0]": Arrow(List(INT), List(INT)),
    "FILTER[EVEN]": Arrow(List(INT), List(INT)),
    "FILTER[ODD]": Arrow(List(INT), List(INT)),
    "MAP[+1]": Arrow(List(INT), List(INT)),
    "MAP[-1]": Arrow(List(INT), List(INT)),
    "MAP[*2]": Arrow(List(INT), List(INT)),
    "MAP[/2]": Arrow(List(INT), List(INT)),
    "MAP[*-1]": Arrow(List(INT), List(INT)),
    "MAP[**2]": Arrow(List(INT), List(INT)),
    "MAP[*3]": Arrow(List(INT), List(INT)),
    "MAP[/3]": Arrow(List(INT), List(INT)),
    "MAP[*4]": Arrow(List(INT), List(INT)),
    "MAP[/4]": Arrow(List(INT), List(INT)),
    "ZIPWITH[+]": Arrow(List(INT), Arrow(List(INT), List(INT))),
    "ZIPWITH[-]": Arrow(List(INT), Arrow(List(INT), List(INT))),
    "ZIPWITH[*]": Arrow(List(INT), Arrow(List(INT), List(INT))),
    "ZIPWITH[min]": Arrow(List(INT), Arrow(List(INT), List(INT))),
    "ZIPWITH[max]": Arrow(List(INT), Arrow(List(INT), List(INT))),
    "SCAN1L[+]": Arrow(List(INT), List(INT)),
    "SCAN1L[-]": Arrow(List(INT), List(INT)),
    "SCAN1L[*]": Arrow(List(INT), List(INT)),
    "SCAN1L[min]": Arrow(List(INT), List(INT)),
    "SCAN1L[max]": Arrow(List(INT), List(INT)),
    # 'MAP': Arrow(Arrow(t0,t1),Arrow(List(t0),List(t1))),
}


__forbidden_patterns = [
    ["COUNT[<0]", "FILTER[<0]"],
    ["COUNT[<0]", "FILTER[>0]"],
    ["COUNT[<0]", "MAP[**2]"],
    ["COUNT[<0]", "MAP[*-1]"],
    ["COUNT[<0]", "MAP[*2]"],
    ["COUNT[<0]", "MAP[*3]"],
    ["COUNT[<0]", "MAP[*4]"],
    ["COUNT[<0]", "REVERSE"],
    ["COUNT[<0]", "SORT"],
    ["COUNT[>0]", "FILTER[<0]"],
    ["COUNT[>0]", "FILTER[>0]"],
    ["COUNT[>0]", "MAP[*-1]"],
    ["COUNT[>0]", "MAP[*2]"],
    ["COUNT[>0]", "MAP[*3]"],
    ["COUNT[>0]", "MAP[*4]"],
    ["COUNT[>0]", "REVERSE"],
    ["COUNT[>0]", "SORT"],
    ["COUNT[EVEN]", "FILTER[EVEN]"],
    ["COUNT[EVEN]", "FILTER[ODD]"],
    ["COUNT[EVEN]", "MAP[**2]"],
    ["COUNT[EVEN]", "MAP[*-1]"],
    ["COUNT[EVEN]", "MAP[*2]"],
    ["COUNT[EVEN]", "MAP[*3]"],
    ["COUNT[EVEN]", "MAP[*4]"],
    ["COUNT[EVEN]", "MAP[+1]"],
    ["COUNT[EVEN]", "MAP[-1]"],
    ["COUNT[EVEN]", "REVERSE"],
    ["COUNT[EVEN]", "SORT"],
    ["COUNT[ODD]", "FILTER[EVEN]"],
    ["COUNT[ODD]", "FILTER[ODD]"],
    ["COUNT[ODD]", "MAP[**2]"],
    ["COUNT[ODD]", "MAP[*-1]"],
    ["COUNT[ODD]", "MAP[*2]"],
    ["COUNT[ODD]", "MAP[*3]"],
    ["COUNT[ODD]", "MAP[*4]"],
    ["COUNT[ODD]", "MAP[+1]"],
    ["COUNT[ODD]", "MAP[-1]"],
    ["COUNT[ODD]", "REVERSE"],
    ["COUNT[ODD]", "SORT"],
    ["FILTER[<0]", "FILTER[<0]"],
    ["FILTER[<0]", "FILTER[>0]"],
    ["FILTER[<0]", "MAP[**2]"],
    ["FILTER[>0]", "FILTER[<0]"],
    ["FILTER[>0]", "FILTER[>0]"],
    ["FILTER[EVEN]", "FILTER[EVEN]"],
    ["FILTER[EVEN]", "FILTER[ODD]"],
    ["FILTER[EVEN]", "MAP[*2]"],
    ["FILTER[EVEN]", "MAP[*4]"],
    ["FILTER[ODD]", "FILTER[EVEN]"],
    ["FILTER[ODD]", "FILTER[ODD]"],
    ["FILTER[ODD]", "MAP[*2]"],
    ["FILTER[ODD]", "MAP[*4]"],
    ["HEAD", "REVERSE"],
    ["HEAD", "SCAN1L[*]"],
    ["HEAD", "SCAN1L[+]"],
    ["HEAD", "SCAN1L[-]"],
    ["HEAD", "SCAN1L[max]"],
    ["HEAD", "SCAN1L[min]"],
    ["HEAD", "SORT"],
    ["LENGTH", "FILTER[<0]"],
    ["LENGTH", "FILTER[>0]"],
    ["LENGTH", "FILTER[EVEN]"],
    ["LENGTH", "FILTER[ODD]"],
    ["LENGTH", "MAP[**2]"],
    ["LENGTH", "MAP[*-1]"],
    ["LENGTH", "MAP[*2]"],
    ["LENGTH", "MAP[*3]"],
    ["LENGTH", "MAP[*4]"],
    ["LENGTH", "MAP[+1]"],
    ["LENGTH", "MAP[-1]"],
    ["LENGTH", "MAP[/2]"],
    ["LENGTH", "MAP[/3]"],
    ["LENGTH", "MAP[/4]"],
    ["LENGTH", "REVERSE"],
    ["LENGTH", "SCAN1L[*]"],
    ["LENGTH", "SCAN1L[+]"],
    ["LENGTH", "SCAN1L[-]"],
    ["LENGTH", "SCAN1L[max]"],
    ["LENGTH", "SCAN1L[min]"],
    ["LENGTH", "SORT"],
    ["MAP[**2]", "MAP[*-1]"],
    ["MAP[*-1]", "MAP[*-1]"],
    ["MAP[*2]", "MAP[*2]"],
    ["MAP[+1]", "MAP[-1]"],
    ["MAP[-1]", "MAP[+1]"],
    ["MAP[/2]", "MAP[*2]"],
    ["MAP[/2]", "MAP[*4]"],
    ["MAP[/2]", "MAP[/2]"],
    ["MAP[/3]", "MAP[*3]"],
    ["MAP[/4]", "MAP[*2]"],
    ["MAP[/4]", "MAP[*4]"],
    ["MAXIMUM", "REVERSE"],
    ["MAXIMUM", "SCAN1L[max]"],
    ["MAXIMUM", "SCAN1L[min]"],
    ["MAXIMUM", "SORT"],
    ["MINIMUM", "REVERSE"],
    ["MINIMUM", "SCAN1L[max]"],
    ["MINIMUM", "SCAN1L[min]"],
    ["MINIMUM", "SORT"],
    ["REVERSE", "REVERSE"],
    ["SCAN1L[max]", "SCAN1L[max]"],
    ["SCAN1L[max]", "SORT"],
    ["SCAN1L[min]", "SCAN1L[min]"],
    ["SORT", "REVERSE"],
    ["SORT", "SCAN1L[max]"],
    ["SORT", "SORT"],
    ["SUM", "REVERSE"],
    ["SUM", "SORT"],
    ["TAIL", "REVERSE"],
    ["TAIL", "SCAN1L[+]"],
    ["TAIL", "SCAN1L[max]"],
    ["TAIL", "SCAN1L[min]"],
    ["TAIL", "SORT"],
]


dsl = DSL(__primitive_types, __forbidden_patterns)
evaluator = DSLEvaluator(__semantics)
evaluator.skip_exceptions.add(OverflowError)
lexicon = list(range(-256, 256 + 1))


def pruned_version(probress_bar: bool = False) -> Tuple[DSL, DSLEvaluator]:
    patterns = [
        "COUNT[<0] ^MAP[*-1],MAP[**2]",
        "COUNT[>0] ^MAP[*-1],MAP[**2]",
        "COUNT[EVEN] ^MAP[+1],MAP[*2]",
        "COUNT[ODD] ^MAP[+1],MAP[*2]",
        "FILTER[EVEN] ^MAP[+1],MAP[*2],FILTER[ODD]",
        "FILTER[ODD] ^MAP[+1],MAP[*2],FILTER[EVEN]",
        "ZIPWITH[+] ^ZIPWITH[+] *",
        "ZIPWITH[*] ^ZIPWITH[*] *",
        "ZIPWITH[min] ^ZIPWITH[min] *",
        "ZIPWITH[max] ^ZIPWITH[max] *",
    ]
    new_syntax, _ = produce_new_syntax_for_constraints(
        __primitive_types, patterns, progress=probress_bar
    )
    new_dsl = DSL(new_syntax, __forbidden_patterns)
    new_semantics = auto_complete_semantics(__primitive_types.keys(), __semantics)
    new_evaluator = DSLEvaluator(new_semantics)
    new_evaluator.skip_exceptions.add(OverflowError)
    return new_dsl, new_evaluator
