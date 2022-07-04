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
from synth.syntax.type_system import UnknownType

VAR = re.compile("'([^' ]*)'")


@dataclass
class ParseContext:
    varno: int = field(default=0)
    variables: Dict[str, Tuple[int, Type]] = field(default_factory=lambda: {})
    modules: Set[str] = field(default_factory=set)
    rest: Set[str] = field(default_factory=set)


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
        assert isinstance(node.ctx, ast.Load)
        value = str(node.id)
        if value in ctx.variables:
            (varno, _) = ctx.variables[value]
            ctx.variables[value] = (varno, guess_type(node.value))
            return Variable(ctx.variables[value][0], ctx.variables[value][1])
        return Primitive(node.id, INT)
    elif isinstance(node, ast.Attribute):
        assert isinstance(node.ctx, ast.Load)
        object = build_program(node.value, ctx)
        # print("attribute:", node.attr, "of", object)
        if isinstance(object, Primitive):
            spec = importlib.util.find_spec(object.primitive)
            if spec is not None:
                ctx.modules.add(object.primitive)
                return Primitive(object.primitive+ "." + node.attr, INT)
            else:
                ctx.rest.add(object.primitive)
                object.type = FunctionType(INT, INT)
                return Function(object, [Primitive(node.attr, INT)])
        return Function(Primitive(node.attr, Arrow(object.type, INT)), [object])
    elif isinstance(node, ast.Expr):
        # print("expression:", node.value)
        return build_program(node.value, ctx)
    elif isinstance(node, ast.Constant):
        value = str(node.value)
        if value in ctx.variables:
            (varno, _) = ctx.variables[value]
            ctx.variables[value] = (varno, guess_type(node.value))
            out = Variable(ctx.variables[value][0], ctx.variables[value][1])
        else:
            # out = Variable(ctx.varno, guess_type(node.value))
            # ctx.variables[node.value] = ctx.varno
            # ctx.varno += 1
            out = Primitive(node.value, guess_type(node.value))
        return out
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
        varno = len(variables)
        variables[content] = (varno, UnknownType())
        intent = intent.replace(var.group(0), f"var{varno}")
    return intent, variables


def try_convert_task(task: Dict[str, str], tasks: TList[Task[NLP]]) -> None:
    intent = task["intent"]
    if task["rewritten_intent"] is not None:
        intent = task["rewritten_intent"].replace("`", "'").replace('"', "'")
    metadata = {"question_id": int(task["question_id"])}
    snippet: str = task["snippet"]
    intent, variables = extract_variables(intent)
    if len(tasks) > 0 and tasks[-1].specification.intent == intent:
        return
    out = try_parse_snippet(snippet, variables)
    if out:
        solution, ctx = out
        vars = list(variables.values())
        vars.sort(key=lambda x: x[0])
        type_request = FunctionType(*[var for var in vars], INT)
        print("Description=", intent)
        # print("Variables=", ctx.variables)
        # print("Type=", solution.type)
        print("Modules=", ctx.modules)
        print("Rest=", ctx.rest)
        print("Solution=", solution)
        print("Original=", task["snippet"])
        tasks.append(Task(solution.type, NLP(intent), solution, metadata))
        print()
    # else:
    # print("FAILED:", intent)


def convert_conala(file: str) -> None:
    filename: str = Path(file).stem
    output_file = f"./{filename}.pickle"

    tasks: List[Task[NLP]] = []
    with open(file) as fd:
        data = json.load(fd)
        for task in tqdm.tqdm(data):
            try_convert_task(task, tasks)

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
