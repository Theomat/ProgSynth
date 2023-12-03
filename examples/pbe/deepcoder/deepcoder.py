from synth.semantic import DSLEvaluator
from synth.syntax import DSL, INT, Arrow, PolymorphicType, List

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
    "LENGTH": len,
    "COUNT[<0]": lambda l: len([x for x in l if x < 0]),
    "COUNT[>0]": lambda l: len([x for x in l if x > 0]),
    "COUNT[EVEN]": lambda l: len([x for x in l if x % 2 == 0]),
    "COUNT[ODD]": lambda l: len([x for x in l if x % 2 == 1]),
    "SUM": sum,
    "TAKE": lambda i: lambda l: l[:i],
    "DROP": lambda i: lambda l: l[i:],
    "SORT": sorted,
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


__forbidden_patterns = {
    "('COUNT[<0]', 0)": {
        "FILTER[<0]",
        "FILTER[>0]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*3]",
        "MAP[*4]",
        "MAP[+1]",
        "REVERSE",
        "SORT",
    },
    "('COUNT[>0]', 0)": {
        "FILTER[<0]",
        "FILTER[>0]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*3]",
        "MAP[*4]",
        "MAP[-1]",
        "REVERSE",
        "SORT",
    },
    "('COUNT[EVEN]', 0)": {
        "FILTER[<0]",
        "FILTER[>0]",
        "FILTER[EVEN]",
        "FILTER[ODD]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*3]",
        "MAP[*4]",
        "MAP[+1]",
        "MAP[-1]",
        "REVERSE",
        "SCAN1L[+]",
        "SORT",
    },
    "('COUNT[ODD]', 0)": {
        "FILTER[<0]",
        "FILTER[>0]",
        "FILTER[EVEN]",
        "FILTER[ODD]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*3]",
        "MAP[*4]",
        "MAP[+1]",
        "MAP[-1]",
        "REVERSE",
        "SCAN1L[+]",
        "SORT",
    },
    "('FILTER[<0]', 0)": {
        "FILTER[<0]",
        "FILTER[>0]",
        "FILTER[EVEN]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*3]",
        "SORT",
    },
    "('FILTER[>0]', 0)": {"FILTER[<0]", "FILTER[>0]", "MAP[**2]", "MAP[*2]", "SORT"},
    "('FILTER[EVEN]', 0)": {
        "FILTER[>0]",
        "FILTER[EVEN]",
        "FILTER[ODD]",
        "MAP[**2]",
        "MAP[*2]",
        "MAP[*4]",
        "SORT",
    },
    "('FILTER[ODD]', 0)": {
        "FILTER[<0]",
        "FILTER[>0]",
        "FILTER[EVEN]",
        "FILTER[ODD]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*3]",
        "MAP[*4]",
        "MAP[+1]",
        "REVERSE",
        "SORT",
    },
    "('HEAD', 0)": {
        "MAP[**2]",
        "MAP[*2]",
        "REVERSE",
        "SCAN1L[*]",
        "SCAN1L[+]",
        "SCAN1L[-]",
        "SCAN1L[max]",
        "SCAN1L[min]",
        "SORT",
    },
    "('LENGTH', 0)": {
        "FILTER[<0]",
        "FILTER[>0]",
        "FILTER[EVEN]",
        "FILTER[ODD]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*3]",
        "MAP[*4]",
        "MAP[+1]",
        "MAP[-1]",
        "MAP[/2]",
        "MAP[/3]",
        "MAP[/4]",
        "REVERSE",
        "SCAN1L[*]",
        "SCAN1L[+]",
        "SCAN1L[-]",
        "SCAN1L[max]",
        "SCAN1L[min]",
        "SORT",
    },
    "('MAP[**2]', 0)": {"FILTER[EVEN]", "MAP[**2]", "MAP[*-1]", "MAP[*2]"},
    "('MAP[*-1]', 0)": {
        "FILTER[<0]",
        "FILTER[EVEN]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*3]",
        "MAP[/2]",
        "SCAN1L[-]",
        "SCAN1L[max]",
        "SCAN1L[min]",
    },
    "('MAP[*2]', 0)": {
        "FILTER[<0]",
        "FILTER[>0]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*3]",
        "MAP[*4]",
        "SCAN1L[+]",
        "SCAN1L[-]",
        "SCAN1L[max]",
        "SCAN1L[min]",
        "SORT",
    },
    "('MAP[*3]', 0)": {"FILTER[>0]", "FILTER[EVEN]", "MAP[**2]", "MAP[*2]", "SORT"},
    "('MAP[*4]', 0)": {
        "FILTER[<0]",
        "FILTER[>0]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*3]",
        "SCAN1L[+]",
        "SCAN1L[-]",
        "SCAN1L[max]",
        "SCAN1L[min]",
        "SORT",
    },
    "('MAP[+1]', 0)": {
        "FILTER[ODD]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[-1]",
        "REVERSE",
        "SCAN1L[max]",
        "SCAN1L[min]",
        "SORT",
    },
    "('MAP[-1]', 0)": {
        "FILTER[EVEN]",
        "FILTER[ODD]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[+1]",
        "REVERSE",
        "SCAN1L[max]",
        "SCAN1L[min]",
        "SORT",
    },
    "('MAP[/2]', 0)": {
        "MAP[**2]",
        "MAP[*2]",
        "MAP[*4]",
        "MAP[/2]",
        "SCAN1L[max]",
        "SCAN1L[min]",
        "SORT",
    },
    "('MAP[/3]', 0)": {
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*3]",
        "MAP[/2]",
        "MAP[/4]",
        "REVERSE",
        "SCAN1L[max]",
        "SCAN1L[min]",
        "SORT",
    },
    "('MAP[/4]', 0)": {
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*4]",
        "MAP[/2]",
        "SCAN1L[max]",
        "SCAN1L[min]",
        "SORT",
    },
    "('MAXIMUM', 0)": {
        "MAP[**2]",
        "MAP[*2]",
        "REVERSE",
        "SCAN1L[max]",
        "SCAN1L[min]",
        "SORT",
    },
    "('MINIMUM', 0)": {
        "MAP[**2]",
        "MAP[*2]",
        "REVERSE",
        "SCAN1L[max]",
        "SCAN1L[min]",
        "SORT",
    },
    "('REVERSE', 0)": {
        "FILTER[<0]",
        "FILTER[>0]",
        "FILTER[EVEN]",
        "MAP[**2]",
        "MAP[*-1]",
        "MAP[*2]",
        "MAP[*3]",
        "MAP[*4]",
        "MAP[/2]",
        "MAP[/4]",
        "REVERSE",
        "SCAN1L[min]",
    },
    "('SCAN1L[*]', 0)": {"MAP[**2]", "MAP[*2]"},
    "('SCAN1L[+]', 0)": {"MAP[**2]", "MAP[*-1]", "MAP[*2]", "MAP[*3]"},
    "('SCAN1L[-]', 0)": {"MAP[**2]", "MAP[*2]", "MAP[*3]"},
    "('SCAN1L[max]', 0)": {"MAP[**2]", "MAP[*2]", "MAP[*3]", "SCAN1L[max]", "SORT"},
    "('SCAN1L[min]', 0)": {
        "MAP[**2]",
        "MAP[*2]",
        "MAP[*3]",
        "SCAN1L[max]",
        "SCAN1L[min]",
    },
    "('SORT', 0)": {"MAP[**2]", "MAP[*2]", "REVERSE", "SCAN1L[max]", "SORT"},
    "('SUM', 0)": {"MAP[**2]", "MAP[*2]", "REVERSE", "SORT"},
    "('TAIL', 0)": {
        "MAP[**2]",
        "MAP[*2]",
        "REVERSE",
        "SCAN1L[+]",
        "SCAN1L[max]",
        "SCAN1L[min]",
        "SORT",
    },
    "('ZIPWITH[*]', 1)": {"ZIPWITH[*]", "ZIPWITH[+]", "ZIPWITH[max]", "ZIPWITH[min]"},
    "('ZIPWITH[+]', 1)": {"ZIPWITH[*]", "ZIPWITH[+]", "ZIPWITH[max]", "ZIPWITH[min]"},
    "('ZIPWITH[max]', 1)": {"ZIPWITH[*]", "ZIPWITH[+]", "ZIPWITH[max]", "ZIPWITH[min]"},
    "('ZIPWITH[min]', 1)": {"ZIPWITH[*]", "ZIPWITH[+]", "ZIPWITH[max]", "ZIPWITH[min]"},
}
dsl = DSL(__primitive_types, __forbidden_patterns)
dsl_raw = DSL(__primitive_types)
evaluator = DSLEvaluator(dsl.instantiate_semantics(__semantics))
evaluator.skip_exceptions.add(OverflowError)
lexicon = list(range(-256, 256 + 1))


# if __name__ == "__main__":
#     import os
#     from synth.pruning import (
#         produce_new_syntax_for_constraints,
#         export_syntax_to_python,
#     )

#     file_path = os.path.realpath(__file__)

#     content = ""
#     with open(file_path) as fd:
#         content = fd.read()

#     new_dsl_file = os.path.join(os.path.dirname(file_path), "deepcoder_pruned.py")
#     patterns = [
#         "ZIPWITH[+] _ ^ZIPWITH[max],ZIPWITH[-],ZIPWITH[+],ZIPWITH[min],ZIPWITH[*]",
#         "ZIPWITH[-] _ ^ZIPWITH[max],ZIPWITH[-],ZIPWITH[+],ZIPWITH[min],ZIPWITH[*]",
#         "ZIPWITH[*] _ ^ZIPWITH[max],ZIPWITH[-],ZIPWITH[+],ZIPWITH[min],ZIPWITH[*]",
#         "ZIPWITH[min] _ ^ZIPWITH[max],ZIPWITH[-],ZIPWITH[+],ZIPWITH[min],ZIPWITH[*]",
#         "ZIPWITH[max] _ ^ZIPWITH[max],ZIPWITH[-],ZIPWITH[+],ZIPWITH[min],ZIPWITH[*]",
#     ]
#     new_syntax, _ = produce_new_syntax_for_constraints(
#         __primitive_types,
#         patterns,
#         forbidden=__forbidden_patterns,
#         progress=True,
#     )
#     with open(new_dsl_file, "w") as fd:
#         p_index = content.index("__primitive_types")
#         fd.write("from synth.semantic.evaluator import auto_complete_semantics\n")
#         fd.write("from synth.syntax import PrimitiveType\n")
#         fd.write(content[:p_index])
#         fd.write(
#             export_syntax_to_python(new_syntax, "__primitive_types").replace(
#                 "int", "INT"
#             )
#         )
#         forbidden_index = content.index("__forbidden", p_index)
#         ev_index = content.index("evaluator = ", forbidden_index)
#         fd.write("\n")
#         fd.write(content[forbidden_index:ev_index])
#         fd.write("auto_complete_semantics(__primitive_types.keys(), __semantics)\n")
#         end_index = content.index('if __name__ == "__main__":', ev_index)
#         fd.write(content[ev_index:end_index])
