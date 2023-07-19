import random
import json
import tqdm

from synth import Task, Dataset, PBE, Example
from typing import Any, Callable, Dict, Tuple, List as TList
from synth.specification import PBEWithConstants
from synth.syntax import (
    STRING,
    FunctionType,
    List,
    Type,
    Function,
    Primitive,
    Program,
    Variable,
    Arrow,
)
from examples.pbe.regexp.type_regex import REGEXP
from examples.pbe.transduction.transduction import dsl, evaluator

import argparse


TRANSDUCTION = "transduction"
TRANSDUCTION_OLD = "transduction_old"

argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Convert flashfill original dataset to ProgSynth format."
)

argument_parser.add_argument(
    type=str,
    dest="file",
    action="store",
    help="Source JSON transduction file to be converted",
)


parsed_parameters = argument_parser.parse_args()


name2type = {p.primitive: p.type for p in dsl.list_primitives}

name_converter = {
    "tail_cst": "split_snd_cst",
    "head_cst": "split_first_cst",
    "head": "split_first",
}


def lcs(u, v):
    # t[(n,m)] = length of longest common string ending at first
    # n elements of u & first m elements of v
    t = {}

    for n in range(len(u) + 1):
        for m in range(len(v) + 1):
            if m == 0 or n == 0:
                t[(n, m)] = 0
                continue

            if u[n - 1] == v[m - 1]:
                t[(n, m)] = 1 + t[(n - 1, m - 1)]
            else:
                t[(n, m)] = 0
    l, n, m = max((l, n, m) for (n, m), l in t.items())
    return u[n - l : n]


delimiters = [".", ",", " ", "(", ")", "-"]
characters = (
    [chr(ord("a") + j) for j in range(26)]
    + [chr(ord("A") + j) for j in range(26)]
    + [str(j) for j in range(10)]
    + ["+"]
)

WORDS = None


def get_lexicon():
    return delimiters + characters


def randomDelimiter():
    return random.choice(delimiters)


def randomCharacter():
    return random.choice(characters)


def randomWord(minimum=1, predicate=None):
    global WORDS
    if WORDS is None:
        tasks = load_tasks()
        observations = {
            "".join(z) for t in tasks for xs, y in t[1] for z in list(xs[:-1]) + [y]
        }

        def splitMany(s, ds):
            if len(ds) == 0:
                return [s]
            d = ds[0]
            ds = ds[1:]
            s = [w for z in s.split(d) for w in splitMany(z, ds) if len(w) > 0]
            return s

        WORDS = {w for o in observations for w in splitMany(o, delimiters)}
        WORDS = list(sorted(list(WORDS)))

    # a disproportionately large fraction of the words have length three
    # the purpose of this is to decrease the number of 3-length words we have
    while True:
        if random.random() > 0.7:
            candidate = random.choice([w for w in WORDS if len(w) >= minimum])
        else:
            candidate = random.choice(
                [w for w in WORDS if len(w) >= minimum and len(w) != 3]
            )
        if predicate is None or predicate(candidate):
            return candidate


def randomWords(ds, minimum=1, lb=2, ub=4):
    words = [
        randomWord(minimum=minimum) for _ in range(random.choice(range(lb, ub + 1)))
    ]
    s = ""
    for j, w in enumerate(words):
        if j > 0:
            s += random.choice(ds)
        s += w
    return s


def make_synthetic_tasks():
    import random

    random.seed(9)

    NUMBEROFEXAMPLES = 4

    problems = []
    # Converts strings into a list of characters depending on the type

    def preprocess(x):
        if isinstance(x, tuple):
            return tuple(preprocess(z) for z in x)
        if isinstance(x, list):
            return [preprocess(z) for z in x]
        if isinstance(x, str):
            return [c for c in x]
        if isinstance(x, bool):
            return x
        assert False

    def problem(n, examples, needToTrain=False):
        task = (n, [(preprocess(x), preprocess(y)) for x, y in examples])
        problems.append(task)

    for d in delimiters:
        problem(
            "drop first word delimited by '%s'" % d,
            [
                ((x,), d.join(x.split(d)[1:]))
                for _ in range(NUMBEROFEXAMPLES)
                for x in [randomWords(d)]
            ],
            needToTrain=True,
        )
        for n in [0, 1, -1]:
            problem(
                "nth (n=%d) word delimited by '%s'" % (n, d),
                [
                    ((x,), x.split(d)[n])
                    for _ in range(NUMBEROFEXAMPLES)
                    for x in [randomWords(d)]
                ],
                needToTrain=True,
            )
    for d1 in delimiters:
        problem(
            "Append two words delimited by '%s'" % (d1),
            [
                ((x, y), x + d1 + y)
                for _ in range(NUMBEROFEXAMPLES)
                for x in [randomWord()]
                for y in [randomWord()]
            ],
            needToTrain=True,
        )

    for n in range(1, 6):
        problem(
            "Drop last %d characters" % n,
            [
                ((x,), x[:-n])
                for _ in range(NUMBEROFEXAMPLES)
                for x in [randomWord(minimum=n)]
            ],
            needToTrain=True,
        )
        if n > 1:
            problem(
                "Take first %d characters" % n,
                [
                    ((x,), x[:n])
                    for _ in range(NUMBEROFEXAMPLES)
                    for x in [randomWord(minimum=n)]
                ],
                needToTrain=True,
            )

    for n in range(len(delimiters)):
        problem(
            "First letters of words (%s)" % ("I" * (1 + n)),
            [
                ((x,), "".join(map(lambda z: z[0], x.split(" "))))
                for _ in range(NUMBEROFEXAMPLES)
                for x in [randomWords(" ")]
            ],
            needToTrain=True,
        )

    for d in delimiters:
        problem(
            "Take first character and append '%s'" % d,
            [((x,), x[0] + d) for _ in range(NUMBEROFEXAMPLES) for x in [randomWord()]],
            needToTrain=True,
        )

    for n in range(len(delimiters)):
        problem(
            "Abbreviate separate words (%s)" % ("I" * (n + 1)),
            [
                ((x, y), "%s.%s." % (x[0], y[0]))
                for _ in range(NUMBEROFEXAMPLES)
                for y in [randomWord()]
                for x in [randomWord()]
            ],
        )
        d = delimiters[n]
        problem(
            "Abbreviate words separated by '%s'" % d,
            [
                ((x + d + y,), "%s.%s." % (x[0], y[0]))
                for _ in range(NUMBEROFEXAMPLES)
                for y in [randomWord()]
                for x in [randomWord()]
            ],
        )

    for n in range(len(delimiters)):
        problem(
            "Append 2 strings (%s)" % ("I" * (n + 1)),
            [
                ((x, y), x + y)
                for _ in range(NUMBEROFEXAMPLES)
                for y in [randomWord()]
                for x in [randomWord()]
            ],
            needToTrain=True,
        )

    for n in range(len(delimiters)):
        w = randomWord(minimum=3)
        problem(
            "Prepend '%s'" % w,
            [((x,), w + x) for _ in range(NUMBEROFEXAMPLES) for x in [randomWord()]],
        )
        w = randomWord(minimum=3)
        problem(
            "Append '%s'" % w,
            [((x,), x + w) for _ in range(NUMBEROFEXAMPLES) for x in [randomWord()]],
        )
        w = randomWord(minimum=3)
        problem(
            "Prepend '%s' to first word" % w,
            [
                ((x + " " + y,), w + x)
                for _ in range(NUMBEROFEXAMPLES)
                for x in [randomWord()]
                for y in [randomWord()]
            ],
        )

    for n in range(1, 6):
        problem(
            "parentheses around a single word (%s)" % ("I" * n),
            [
                ((w,), "(%s)" % w)
                for _ in range(NUMBEROFEXAMPLES)
                for w in [randomWord()]
            ],
        )
    problem(
        "parentheses around first word",
        [
            ((w + " " + s,), "(%s)" % w)
            for _ in range(NUMBEROFEXAMPLES)
            for w in [randomWord()]
            for s in [randomWords(" ")]
        ],
    )
    problem(
        "parentheses around second word",
        [
            ((s,), "(%s)" % (s.split(" ")[1]))
            for _ in range(NUMBEROFEXAMPLES)
            for s in [randomWords(" ")]
        ],
    )

    for n in range(7):
        w = randomWord(minimum=3)
        problem(
            "ensure suffix `%s`" % w,
            [
                ((s + (w if f else ""),), s + w)
                for _ in range(NUMBEROFEXAMPLES)
                for s in [randomWords(" ")]
                for f in [random.choice([True, False])]
            ],
        )

    return [add_constants_to_task(p) for p in problems]


def load_tasks(folder="flashfill_dataset"):
    """
    Processes sygus benchmarks into task objects
    For these benchmarks, all of the constant strings are given to us.
    In a sense this is cheating
    Returns (tasksWithoutCheating, tasksWithCheating).
    NB: Results in paper are done without "cheating"
    """
    import os
    from sexpdata import loads, Symbol

    def findStrings(s):
        if isinstance(s, list):
            return [y for x in s for y in findStrings(x)]
        if isinstance(s, str):
            return [s]
        return []

    def explode(s):
        return [c for c in s]

    tasks = []
    cheatingTasks = []
    for f in os.listdir(folder):
        if not f.endswith(".sl"):
            continue
        with open(folder + "/" + f, "r") as handle:
            message = "(%s)" % (handle.read())

        expression = loads(message)

        constants = []
        name = f
        examples = []
        declarative = False
        for e in expression:
            if len(e) == 0:
                continue
            if e[0] == Symbol("constraint"):
                e = e[1]
                assert e[0] == Symbol("=")
                inputs = e[1]
                assert inputs[0] == Symbol("f")
                inputs = inputs[1:]
                output = e[2]
                examples.append((inputs, output))
            elif e[0] == Symbol("synth-fun"):
                if e[1] == Symbol("f"):
                    constants += findStrings(e)
                else:
                    declarative = True
                    break
        if declarative:
            continue

        examples = list({(tuple(xs), y) for xs, y in examples})

        task = (name, [(tuple(map(explode, xs)), explode(y)) for xs, y in examples])
        cheat = task

        tasks.append(task)
        cheatingTasks.append(cheat)
    print(name)
    tasks = [
        (name, [(["".join(x) for x in xs] + [None], "".join(y)) for xs, y in examples])
        for name, examples in tasks
    ]
    return [add_constants_to_task(p) for p in tasks]


def add_constants_to_task(task):
    name, examples = task
    constants = []
    if type(examples[0][-1]) == str:
        guesses = {}
        N = 10
        T = 2
        for n in range(min(N, len(examples))):
            for m in range(n + 1, min(N, len(examples))):
                y1 = examples[n][1]
                y2 = examples[m][1]
                l = "".join(lcs(y1, y2))
                if len(l) > 2:
                    guesses[l] = guesses.get(l, 0) + 1

    # Custom addition to constants
    # We add all characters that are in all input or in all outputs
    constants_out = []
    all_i = [
        list(examples[0][0][i])
        for i in range(len(examples[0][0]))
        if examples[0][0][i] is not None
    ]

    all_o = list(examples[0][1])
    separator = "#"
    for i, o in examples:
        for index in range(len(all_o[:])):
            if not all_o[index] in o:
                all_o[index] = separator
        for num in range(len(all_i)):
            for index in range(len(all_i[num][:])):
                if all_i[num][index] not in i[num]:
                    all_i[num][index] = separator

    all_o.insert(0, separator)
    all_o.append(separator)
    word = ""
    for l in all_i[0]:
        if l == separator:
            if len(word) > 1 or word in delimiters:
                constants.append(word)
            word = ""
        else:
            word += l
    for l in all_o:
        if l == separator:
            if len(word) > 1 or word in delimiters:
                constants_out.append(word)
            word = ""
        else:
            word += l
    return name, examples, list(set(constants)), list(set(constants_out))


def flashfill2json(tasks, output: str = "flashfill.json"):
    with open(output, "w") as fd:
        result = []
        for t in tasks:
            obj = {"program": None}
            obj["metadata"] = {"name": t[0]}
            e = []
            examples = t[1]
            for i, o in examples:
                i_str = ["".join(j) for j in i]
                o_str = "".join(o)
                e.append({"inputs": i_str, "output": o_str})
            obj["examples"] = e
            obj["constants_in"] = t[2]
            obj["constants_out"] = t[3]
            result.append(obj)
        json.dump(result, fd, indent=4)


def __flashfill_str2prog__(s: str) -> Tuple[Program, Type]:
    if s is None:
        return None, Arrow(STRING, STRING)
    parts = s.split("|")
    stack: TList[Program] = []
    var: int = 0
    type_stack: TList[Type] = []
    for part in parts:
        subparts = part.split(",")
        name = subparts.pop(0)
        if name == "STRING":
            stack.append(Variable(var, STRING))
            var += 1
            type_stack.append(STRING)
            continue
        if name in name_converter:
            name = name_converter[name]
        # primitives that serve as constants
        if name in ["cst_in", "cst_out", "W", "$", ".", "epsilon"]:
            primitive = Primitive(name, name2type[name])
            stack.append(primitive)
        else:  # other primitives are functions, we want to add their type
            targets = [int(x) for x in subparts]
            arguments = [stack[x] for x in targets]
            primitive = Primitive(name, name2type[name])
            stack.append(Function(primitive, arguments))
    type_stack.append(stack[-1].type)
    type_request = FunctionType(*type_stack)
    return stack[-1], type_request


def __convert__(load: Callable[[], Dataset[PBEWithConstants]], name: str) -> None:
    tasks = load()
    tasks.save(name)
    sols = sum(1 for t in tasks if t.solution)
    print(f"Converted {len(tasks)} tasks {sols / len(tasks):.0%} containing solutions")
    # Integrity check
    for task in tqdm.tqdm(tasks, desc="integrity check"):
        if task.solution is None:
            print("Unsolved task: ", task.metadata["name"])
            continue
        for ex in task.specification.examples:
            found = False
            constants_in = task.specification.constants_in
            constants_in.append("")
            constants_out = task.specification.constants_out
            constants_out.append("")
            for cons_in in constants_in:
                for cons_out in constants_out:
                    if found:
                        break
                    # print(evaluator.eval_with_constant(task.solution, ex.inputs, cons_in, cons_out), " vs ", ex.output)
                    found = (
                        evaluator.eval_with_constant(
                            task.solution, ex.inputs, cons_in, cons_out
                        )
                        == ex.output
                    )

            if not found:
                print(task.metadata["name"], "FAILED")
            assert found


def convert_flashfill(input: str, name2type: Dict[str, Any]):
    def load():
        tasks: TList[Task[PBE]] = []
        with open(input, "r") as fd:
            raw_tasks: TList[Dict[str, Any]] = json.load(fd)
            for raw_task in tqdm.tqdm(raw_tasks, desc="converting"):
                program: str = raw_task["program"]
                raw_examples: TList[Dict[str, Any]] = raw_task["examples"]
                inputs = [raw_example["inputs"] for raw_example in raw_examples]
                outputs: TList = [raw_example["output"] for raw_example in raw_examples]
                constants_in: TList[str] = raw_task["constants_in"]
                constants_out: TList[str] = raw_task["constants_out"]
                metadata: Dict[str, Any] = raw_task["metadata"]
                name: str = metadata["name"]
                prog, type_request = __flashfill_str2prog__(program)
                examples = [
                    Example(inp, out)
                    for inp, out in zip(inputs, outputs)
                    if out is not None
                ]
                if len(examples) < len(inputs):
                    continue
                tasks.append(
                    Task[PBEWithConstants](
                        type_request,
                        PBEWithConstants(examples, constants_in, constants_out),
                        prog,
                        {"name": name},
                    )
                )
        return Dataset(tasks, metadata={"dataset": "flashfill", "source:": input})

    __convert__(load, "flashfill.pickle")


if __name__ == "__main__":
    # challenge = load_tasks("flashfill_dataset")
    # tasks = make_synthetic_tasks()
    # print(len(tasks), "synthetic tasks")

    # flashfill2json(tasks, parsed_parameters.file)
    convert_flashfill(parsed_parameters.file, name2type)
