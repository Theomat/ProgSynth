import argparse
from typing import Any, Dict, List, Optional, Tuple

import json

from synth.syntax.grammars.grammar import DerivableProgram
from synth.syntax import DFTA, PrimitiveType, Type, FunctionType, Primitive
from synth.pruning.constraints import add_dfta_constraints

from logics import LIA, NIA, LRA, NRA, BV

from parsing.ast import (
    GroupedRuleList,
    Grammar,
    GrammarTermKind,
    IdentifierTerm,
    LiteralTerm,
    FunctionApplicationTerm,
    Term,
)
from parsing.resolution import FunctionDescriptor, SortDescriptor, SymbolTable
from parsing.symbol_table_builder import SymbolTableBuilder
from parsing.utilities import Location


parser = argparse.ArgumentParser(description="Sharpens a SyGuS grammar")
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="spec.sl",
    help="output file (default: spec.sl)",
)
parser.add_argument(
    "input_file",
    type=argparse.FileType("r"),
    help='Path to a SyGuS input file (or stdin if "-")',
)
parser.add_argument(
    "sharpening_file",
    type=argparse.FileType("r"),
    help="Path to a sharpening JSON input file",
)
parser.add_argument("--v1", action="store_true", help="Use SyGuS V1 specification")

parameters = parser.parse_args()
output_file: str = parameters.output
use_v1: bool = parameters.v1

if use_v1:
    from parsing.v1.parser import SygusV1Parser

    parser = SygusV1Parser()
else:
    from parsing.v2.parser import SygusV2Parser

    parser = SygusV2Parser()
content: str = parameters.input_file.read()
program = parser.parse(content)
symbol_table = SymbolTableBuilder.run(program)
sharpening_rules = json.load(parameters.sharpening_file)

print(f"Found {len(sharpening_rules)} sharpening rules!")

__logic_map = {"LIA": LIA, "NIA": NIA, "LRA": LRA, "NRA": NRA, "BV": BV}


def type_of_symbol(symbol_table: SymbolTable, symbol: str) -> Type:
    descriptor = symbol_table.lookup_symbol(symbol)
    return PrimitiveType(descriptor.symbol_sort.identifier.symbol)


def term2str(term: Term, symbol_table: SymbolTable) -> Tuple[Type, str]:
    if isinstance(term, IdentifierTerm):
        return (
            type_of_symbol(symbol_table, term.identifier.symbol),
            term.identifier.symbol,
        )
    elif isinstance(term, LiteralTerm):
        return (
            PrimitiveType(term.literal.literal_kind.name),
            term.literal.literal_value,
        )
    raise NotImplementedError()


def to_dfta(
    symbol_table: SymbolTable, val: FunctionDescriptor
) -> DFTA[Tuple[Type, str], DerivableProgram]:
    grammar: Grammar = val.synthesis_grammar
    rules: Dict[
        Tuple[
            DerivableProgram,
            Tuple[Tuple[Type, str], ...],
        ],
        Tuple[Type, str],
    ] = {}
    if grammar is None:
        assert (
            symbol_table.logic_name in __logic_map
        ), f"Set logic {symbol_table.logic_name} not supported!"
        dsl = __logic_map[symbol_table.logic_name]

        def type2state(t: Type) -> Tuple[Type, str]:
            return (t, "Start" + str(t))

        for primitive in dsl.list_primitives:
            rules[
                (primitive, tuple(map(type2state, primitive.type.arguments())))
            ] = type2state(primitive.type.returns())

        # Add variables
        for x, y in zip(val.argument_names, val.argument_sorts):
            var_type = PrimitiveType(y.identifier.symbol)
            rules[(Primitive(x, var_type), ())] = type2state(var_type)

        # Add fictional constant
        all_const_types = set()
        for primitive in dsl.list_primitives:
            for base in primitive.type.decompose_type()[0]:
                if "Const" in base.type_name and base.type_name not in all_const_types:
                    all_const_types.add(base.type_name)
                    rules[(Primitive(f"keep@{base}", base), ())] = type2state(base)

    else:
        # Now create rules
        for S, rule in grammar.grouped_rule_lists.items():
            r: GroupedRuleList = rule
            for out in r.expansion_rules:
                if out.grammar_term_kind == GrammarTermKind.BINDER_FREE:
                    if isinstance(out.binder_free_term, FunctionApplicationTerm):
                        f = out.binder_free_term.function_identifier.symbol
                        args = tuple(
                            map(
                                lambda x: term2str(x, symbol_table),
                                out.binder_free_term.arguments,
                            )
                        )
                        fun = Primitive(
                            f,
                            FunctionType(
                                *[x[0] for x in args], type_of_symbol(symbol_table, S)
                            ),
                        )
                        rules[(fun, args)] = (type_of_symbol(symbol_table, S), S)
                    else:
                        t, name = term2str(out.binder_free_term, symbol_table)
                        rules[(Primitive(str(name), t), ())] = (
                            type_of_symbol(symbol_table, S),
                            S,
                        )
    finals = set()
    s: SortDescriptor = val.range_sort
    out_type = PrimitiveType(s.identifier.symbol)

    for _, state in rules.items():
        if state[0] == out_type:
            finals.add(state)
    dfta = DFTA(rules, finals)
    return dfta


def get_root_tag(x: Any) -> Any:
    if isinstance(x, Tuple):
        return get_root_tag(x[0])
    return x


def get_state_name(x: Any) -> str:
    if isinstance(x, str):
        return x
    elif isinstance(x, Tuple):
        for el in x:
            out = get_state_name(el)
            if out:
                return out
    else:
        return ""


def next_tag(tag: str, count: List[int]) -> str:
    if count:
        out = tag + "".join(map(str, count))
        count[0] += 1
        i = 0
        while count[i] > 9:
            count[i] = 0
            i += 1
            if i < len(count):
                count[i] += 1
            else:
                count.append(0)
        return out
    else:
        count.append(0)
        return tag


def from_dfta(dfta: DFTA[Tuple[Tuple[Type, Any], ...], DerivableProgram]) -> str:
    out = ""
    rules: Dict[Tuple[Tuple[Type, Any], ...], List[str]] = {}
    # Convert states to non terminals
    used = set()
    state2name = {}
    for state in dfta.states:
        if state not in state2name:
            base_tag = get_state_name(state)
            tag = base_tag
            count = []
            while tag in used:
                tag = next_tag(base_tag, count)
            used.add(tag)
            state2name[state] = tag
            rules[state] = []
    # Declare non terminals
    out += "\t("
    for state, name in sorted(state2name.items(), key=lambda x: x[1]):
        t = get_root_tag(state).type_name.replace("Const", "")
        out += f"({name} {t}) "
    out += ")\n\n"
    # Make derivation rule
    for (letter, args), dst in dfta.rules.items():
        name = state2name[dst]
        derivation = str(letter)
        if args:
            derivation += " " + " ".join(map(lambda x: state2name[x], args))
            derivation = f"({derivation})"
        elif isinstance(letter, Primitive) and letter.primitive.startswith("keep@"):
            real_type = letter.primitive.replace("Const", "")[len("keep@") :]
            rules[dst].append(f"(Constant {real_type})")
            continue
        rules[dst].append(derivation)

    for state, derivations in rules.items():
        t = get_root_tag(state).type_name.replace("Const", "")
        der = " ".join(derivations)
        out += f"\t(({state2name[state]} {t} ({der})))\n"
    return out


def capture_text(src: str, start: Location, end: Location) -> str:
    lines = src.splitlines()
    relevant = lines[start.line : end.line]
    relevant[0] = relevant[0][start.col - 1 :]
    relevant[-1] = relevant[-1][: end.col]
    return "\n".join(relevant)


def before(a: Optional[Location], b: Optional[Location]) -> bool:
    if a is None:
        return False
    elif b is None:
        return False
    return a.line < b.line or a.col < b.col


if symbol_table.logic_name == "BV":
    print("Bit Vector is not yet fully supported!")


exchanges = []
for key, val in symbol_table.synth_functions.items():
    dfta = to_dfta(symbol_table, val)

    to_replace_with = from_dfta(add_dfta_constraints(dfta, sharpening_rules))

    s: SortDescriptor = val.range_sort
    args = []
    for x, y in zip(val.argument_names, val.argument_sorts):
        args.append(f"({x} {y.identifier})")
    sargs = " ".join(args)
    prefix = f"(synth-fun {key.symbol} ({sargs}) {s.identifier.symbol}"

    grammar: Grammar = val.synthesis_grammar
    if grammar is None:
        to_be_replaced = prefix
        # Special case for Bit Vectors
        if symbol_table.logic_name == "BV":
            to_replace_with = to_replace_with.replace("BitVector32", "(_ BitVec 32)")
    else:
        start = grammar.start_location
        for (name, t) in grammar.nonterminals:
            loc = t.start_location
            if not before(start, loc):
                start = loc
        to_be_replaced = capture_text(content, start, grammar.end_location)

    exchanges.append((to_be_replaced, prefix + "\n" + to_replace_with))


# Replace text now
for repl, new in exchanges:
    content = content.replace(repl, "\n" + new, 1)


with open(output_file, "w") as fd:
    fd.write(content)
print("Saved new specification file to", output_file)
