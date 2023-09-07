import argparse
from typing import Dict, Tuple

import tqdm

from synth import Dataset, PBE
from synth.utils import chrono
from synth.syntax import CFG, DFTA

from parsing.ast import (
    GroupedRuleList,
    Grammar,
    GrammarTermKind,
    IdentifierTerm,
    LiteralTerm,
    FunctionApplicationTerm,
    Term,
)
from parsing.symbol_table_builder import SymbolTableBuilder
from parsing.resolution import SymbolTable


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
    help="Path to a sharpening input file",
)
parser.add_argument("--v1", action="store_true", help="Use SyGuS V1 specification")

parameters = parser.parse_args()
output_file: str = parameters.output
use_v1: bool = parameters.v1

if use_v1:
    from parsing.v1.parser import SygusV1Parser
    from parsing.v1.printer import SygusV1ASTPrinter as printer

    parser = SygusV1Parser()
else:
    from parsing.v2.parser import SygusV2Parser
    from parsing.v2.printer import SygusV2ASTPrinter as printer

    parser = SygusV2Parser()

program = parser.parse(parameters.input_file.read())
symbol_table = SymbolTableBuilder.run(program)


def term2str(term: Term) -> str:
    if isinstance(term, IdentifierTerm):
        return term.identifier.symbol
    elif isinstance(term, LiteralTerm):
        return term.literal.literal_value
    raise NotImplementedError()


def to_dfta(symbol_table: SymbolTable):
    key, val = list(symbol_table.synth_functions.items())[0]
    grammar: Grammar = val.synthesis_grammar
    rules: Dict[
        Tuple[
            str,
            Tuple[str, ...],
        ],
        str,
    ] = {}
    # Copy non terminals
    non_terminals = {}
    for nt in grammar.nonterminals:
        non_terminals[nt] = nt[0]
    # Now create rules
    for S, rule in grammar.grouped_rule_lists.items():
        r: GroupedRuleList = rule
        # print(
        #     "\tS:",
        #     S,
        #     "=",
        #     r.head_symbol_sort_descriptor,
        #     "(",
        #     r.head_symbol_sort_expression,
        #     ") =>",
        # )
        for out in r.expansion_rules:
            if out.grammar_term_kind == GrammarTermKind.BINDER_FREE:
                if isinstance(out.binder_free_term, FunctionApplicationTerm):
                    f = out.binder_free_term.function_identifier.symbol
                    args = tuple(map(term2str, out.binder_free_term.arguments))
                    rules[(f, args)] = S
                else:
                    rules[(term2str(out.binder_free_term), ())] = S
            # else:
            #     print("\t\t [", out.grammar_term_kind, "] =>", out.sort_expression)
    finals = set()
    dfta = DFTA(rules, finals)
    print(dfta)


to_dfta(symbol_table)
