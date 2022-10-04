from synth.syntax import FunctionType, INT, PrimitiveType, BOOL

BLOCK = PrimitiveType("block")


syntax = {
    "ifX": FunctionType(BOOL, BLOCK, BLOCK),
    "ifY": FunctionType(BOOL, BLOCK, BLOCK),
    "elifX": FunctionType(BLOCK, BLOCK, BLOCK),
    "elifY": FunctionType(BLOCK, BLOCK, BLOCK),
    "and": FunctionType(BOOL, BOOL, BOOL),
    "or": FunctionType(BOOL, BOOL, BOOL),
    "not": FunctionType(BOOL, BOOL),
    "+": FunctionType(INT, INT, INT),
    "*": FunctionType(INT, INT, INT),
    "/": FunctionType(INT, INT, INT),
    "-": FunctionType(INT, INT, INT),
    "<=": FunctionType(INT, INT, BOOL),
    ">=": FunctionType(INT, INT, BOOL),
    "==": FunctionType(INT, INT, BOOL),
    "!=": FunctionType(INT, INT, BOOL),
    "%": FunctionType(INT, INT, INT),
    "0": INT,
    "1": INT,
    "2": INT,
    "3": INT,
    "4": INT,
    "3x1": BLOCK,
    "1x3": BLOCK,
    "EMPTY": BLOCK,
}

sketch = "(ifY _ _)"

constraints = [
    "(ifY >^(var1) ifX,elifY,EMPTY,1x3,3x1)",
    "(elifY ifY elifY,EMPTY,1x3,3x1)",
    "(ifX _ elifX,3x1,1x3,EMPTY)",
    "(elifX ifX elifX,EMPTY,1x3,3x1)",
    "(and ^and _)",
    "(or ^or,and ^and)",
    "(not ^not,==,!=)",
    "(elifY ifY elifY,EMPTY,1x3,3x1)",
    "(+ ^+,0 ^*,0)",
    "(- _ ^0)",
    "(* ^*,2,1,0 ^+,2,1,0)",
    "(/ _ ^*)",
    "(% ^0,1 ^0,1)",
]

dfta_constraints = ["(% #(var0,var1)>=1 _)", "(<=,>=,!= #(var0,var1)>=1 _)"]
