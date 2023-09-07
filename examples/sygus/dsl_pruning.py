import argparse

import tqdm

from synth import Dataset, PBE
from synth.utils import chrono
from synth.syntax import CFG

from parsing.symbol_table_builder import SymbolTableBuilder


parser = argparse.ArgumentParser(
    description="Sharpens a SyGuS grammar"
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="spec.sl",
    help="output file (default: spec.sl)",
)
parser.add_argument('input_file', type=argparse.FileType('r'),
        help='Path to a SyGuS input file (or stdin if "-")')
parser.add_argument('sharpening_file', type=argparse.FileType('r'),
        help='Path to a sharpening input file')
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
print(printer.run(program, symbol_table, vars(parameters)))