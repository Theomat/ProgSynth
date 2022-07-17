from synth.syntax.type_system import BOOL, INT, FunctionType, PrimitiveType
from synth.pruning.type_constraints.utils import clean, export_syntax_to_python

BOOL_0 = PrimitiveType("bool@0")
syntax = {
    "+": FunctionType(INT, INT, INT),
    "-": FunctionType(INT, INT, INT),
    "*": FunctionType(INT, INT, INT),
    "and@0": FunctionType(BOOL, BOOL, BOOL_0),
    "or@0": FunctionType(BOOL, BOOL, BOOL_0),
    "not@0": FunctionType(BOOL, BOOL_0),
    "<=@0": FunctionType(INT, INT, BOOL_0),
    "==@0": FunctionType(INT, INT, BOOL_0),
    "<=": FunctionType(INT, INT, BOOL),
    "==": FunctionType(INT, INT, BOOL),
    "and": FunctionType(BOOL, BOOL, BOOL),
    "or": FunctionType(BOOL, BOOL, BOOL),
    "not": FunctionType(BOOL, BOOL),
    "ite": FunctionType(BOOL, INT, INT, INT),
    "0": INT,
    "1": INT,
    "2": INT,
}


def test_export() -> None:
    out = export_syntax_to_python(syntax, varname="koala")
    out = out.replace("int", "INT")
    out = out.replace("bool", "BOOL")
    out = out.replace("BOOL@0", "bool@0")
    imports = "from synth.syntax.type_system import BOOL, INT, FunctionType, PrimitiveType, Arrow\n"
    exec(imports + out)
    koala = eval(out[out.rfind("= {") + 2 :])
    assert koala == syntax


def test_clean() -> None:
    new_syntax = {k: v for k, v in syntax.items()}
    clean(new_syntax, None)
    for P in syntax.keys():
        if "@" in P:
            assert P not in new_syntax
