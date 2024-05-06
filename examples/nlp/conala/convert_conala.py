import ast
import json
import re
import importlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple, List as TList
from pathlib import Path

import tqdm

from synth import Task, Dataset
from synth.specification import NLP
from synth.syntax import (
    INT,
    FunctionType,
    List,
    Function,
    Primitive,
    Program,
    Variable,
    Arrow,
    Type,
    guess_type,
)

from conala import dsl, evaluator
from synth.syntax.type_system import STRING, PolymorphicType, UnknownType

VAR = re.compile("'([^' ]*)'")

KNOWN_TYPES = {
    "string": STRING,
    "list": List(PolymorphicType("any")),
    "integer": INT,
}


@dataclass
class ParseContext:
    varno: int = field(default=0)
    variables: Dict[str, Tuple[int, Type]] = field(default_factory=lambda: {})
    modules: Set[str] = field(default_factory=set)
    rest: Set[str] = field(default_factory=set)

    def variable_by_no(self, no: int) -> Optional[Tuple[str, Type]]:
        for name, (varno, t) in self.variables.items():
            if varno == no:
                return name, t


def try_parse_variable(value: str, ctx: ParseContext) -> Optional[Variable]:
    if value in ctx.variables:
        (varno, var_type) = ctx.variables[value]
        return Variable(varno, var_type)
    return None


def build_program(node: ast.AST, ctx: ParseContext) -> Program:
    if isinstance(node, ast.Call):
        func = build_program(node.func, ctx)
        if len(node.args) == 0:
            return func
        args = [build_program(arg, ctx) for arg in node.args]
        correct_type = FunctionType(*[arg.type for arg in args], INT)
        if isinstance(func, Function):
            func.function.type = FunctionType(
                *[arg.type for arg in func.arguments], correct_type
            )
            func = Function(func.function, func.arguments)
        elif isinstance(func, Primitive):
            func.type = correct_type
        # print("call:", func, "\n\t", args, "\n\ttype:", correct_type, [arg.type for arg in args])
        assert isinstance(
            func.type, Arrow
        ), f"func={func}, type={func.type} type's type={type(func.type)}"
        if len(node.keywords) > 0:
            raise ValueError
        return Function(func, args)
    elif isinstance(node, ast.Name):
        # This is basically module names and variables
        assert isinstance(node.ctx, ast.Load)
        value = str(node.id)
        var_prog = try_parse_variable(value, ctx)
        return var_prog if var_prog else Primitive(node.id, INT)
    elif isinstance(node, ast.Attribute):
        assert isinstance(node.ctx, ast.Load)
        object = build_program(node.value, ctx)
        # print("attribute:", node.attr, "of", object)
        if isinstance(object, Primitive):
            spec = importlib.util.find_spec(object.primitive)
            if spec is not None:
                ctx.modules.add(object.primitive)
                return Primitive(object.primitive + "." + node.attr, INT)
            else:
                ctx.rest.add(object.primitive)
                object.type = FunctionType(INT, INT)
                return Function(object, [Primitive(node.attr, INT)])
        elif isinstance(object, Variable):
            res = ctx.variable_by_no(object.variable)
            assert res is not None
            _, vartype = res
            return Function(
                Primitive(f"{vartype}.{node.attr}", Arrow(object.type, INT)), [object]
            )
        return Function(Primitive(node.attr, Arrow(object.type, INT)), [object])
    elif isinstance(node, ast.Expr):
        return build_program(node.value, ctx)
    elif isinstance(node, ast.Constant):
        value = str(node.value)
        var_prog = try_parse_variable(value, ctx)
        if var_prog is None:
            # out = Variable(ctx.varno, guess_type(node.value))
            # ctx.variables[node.value] = ctx.varno
            # ctx.varno += 1
            out = Primitive(node.value, guess_type(node.value))
            assert False
        else:
            return var_prog
    elif isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        elements = [build_program(el, ctx) for el in node.elts]
        if not all(isinstance(el, Primitive) for el in elements):
            raise ValueError
        return Function(
            Primitive(
                "list", FunctionType(*[el.type for el in elements], INT), elements
            )
        )
    elif isinstance(node, ast.Subscript):
        assert isinstance(node.ctx, ast.Load)
        object = build_program(node.value, ctx)
        # print("SLICE:", node.slice)
        slice = build_program(node.slice, ctx)
        # raise ValueError
        elements = [slice, object]
        return Function(
            Primitive(
                "slice", FunctionType(*[el.type for el in elements], INT), elements
            )
        )
    else:
        # print("CANNOT BUILD:", node)
        raise ValueError


def try_parse_snippet(
    snippet: str, variables: Dict[str, int]
) -> Optional[Tuple[Program, ParseContext]]:
    tree = ast.parse(snippet)
    lines = tree.body
    if len(lines) > 1:
        return None
    content = lines[0]
    ctx = ParseContext(len(variables), variables)
    try:
        return build_program(content, ctx), ctx
    except:
        return None


def extract_variables(intent: str) -> Tuple[str, Dict[str, Tuple[int, Type]]]:
    variables = {}
    for var in re.finditer(VAR, intent):
        content = var.group(1).encode().decode("unicode_escape")

        start_pos = var.start() - 1
        word_before = intent[intent.rfind(" ", 0, start_pos) + 1 : start_pos]
        var_type = KNOWN_TYPES.get(word_before, UnknownType())
        varno = len(variables)
        variables[content] = (varno, var_type)
        intent = intent.replace(var.group(0), f"var{varno}")
    return intent, variables


def try_convert_task(task: Dict[str, str], tasks: TList[Task[NLP]]) -> None:
    intent = task["intent"]
    if task["rewritten_intent"] is None:
        return
    intent = task["rewritten_intent"].replace("`", "'").replace('"', "'")
    metadata = {"question_id": int(task["question_id"])}
    snippet: str = task["snippet"]
    try:
        intent, variables = extract_variables(intent)
    except UnicodeDecodeError:
        return
    if len(tasks) > 0 and tasks[-1].specification.intent == intent:
        return
    out = try_parse_snippet(snippet, variables)
    if out and len(out[1].rest) == 0:
        solution, ctx = out
        vars = list(variables.values())
        vars.sort(key=lambda x: x[0])
        if any(isinstance(var[1], UnknownType) for var in vars):
            return
        type_request = FunctionType(*[var[1] for var in vars], INT)
        print("OG Description=", task["rewritten_intent"])
        print("Description=", intent)
        print("TR=", type_request)
        print("Variables=", ctx.variables)
        print("Type=", solution.type)
        # print("Modules=", ctx.modules)
        # print("Rest=", ctx.rest)
        print("Solution=", solution)
        print("Original=", task["snippet"])
        print()
        tasks.append(Task(solution.type, NLP(intent), solution, metadata))
    # else:
    # print("FAILED:", intent)


def convert_conala(file: str) -> None:
    filename: str = Path(file).stem
    output_file = f"./{filename}.pickle"

    # 1st task processing
    # Crude parsing of programs
    tasks: List[Task[NLP]] = []
    with open(file) as fd:
        data = json.load(fd)
        for task in tqdm.tqdm(data):
            try_convert_task(task, tasks)

    # 2nd task processing
    # Remove constants out
    primitives = set()
    for task in tasks:
        sol = task.solution
        assert sol is not None
        for P in sol.depth_first_iter():
            if isinstance(P, Primitive):
                primitives.add(P.primitive)
    print("Primitives:", primitives)
    dataset = Dataset(tasks)
    dataset.save(output_file)
    print(
        "Successfully saved converted dataset file as",
        output_file,
        "with",
        len(tasks),
        "tasks",
    )


if __name__ == "__main__":
    import argparse

    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert and filter CoNaLa original dataset to ProgSynth format."
    )

    argument_parser.add_argument(
        type=str,
        dest="file",
        action="store",
        help="Source JSON CoNaLa file to be converted",
    )
    parsed_parameters = argument_parser.parse_args()
    convert_conala(parsed_parameters.file)
