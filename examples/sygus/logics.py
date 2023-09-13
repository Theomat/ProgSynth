"""
    Contains the DSL definitions for most SyGuS logics
"""
from synth.syntax import auto_type, DSL


__lia_syntax = auto_type(
    {
        "+": "Int -> Int -> Int",
        "-": "Int -> Int -> Int",
        "*": "constInt -> Int -> Int",
        "*": "Int -> ConstInt -> Int",
        "div": "Int -> ConstInt -> Int",
        "mod": "Int -> ConstInt -> Int",
        "abs": "Int -> Int",
        "ite": "Bool -> Int -> Int -> Int",
        "=": "Int -> Int -> Bool",
        "<=": "Int -> Int -> Bool",
        ">=": "Int -> Int -> Bool",
        "<": "Int -> Int -> Bool",
        ">": "Int -> Int -> Bool",
    }
)

LIA = DSL(__lia_syntax)


__nia_syntax = auto_type(
    {
        "+": "Int -> Int -> Int",
        "-": "Int -> Int -> Int",
        "*": "Int -> Int -> Int",
        "*": "Int -> Int -> Int",
        "div": "Int -> Int -> Int",
        "mod": "Int -> Int -> Int",
        "abs": "Int -> Int",
        "ite": "Bool -> Int -> Int -> Int",
        "=": "Int -> Int -> Bool",
        "<=": "Int -> Int -> Bool",
        ">=": "Int -> Int -> Bool",
        "<": "Int -> Int -> Bool",
        ">": "Int -> Int -> Bool",
    }
)

NIA = DSL(__nia_syntax)

__lra_syntax = auto_type(
    {
        "+": "Real -> Real -> Real",
        "-": "Real -> Real -> Real",
        "*": "ConstReal -> Real -> Real",
        "*": "Real -> ConstReal -> Real",
        "/": "Real -> ConstReal -> Real",
        "ite": "Bool -> Real -> Real -> Real",
        "=": "Real -> Real -> Bool",
        "<=": "Real -> Real -> Bool",
        ">=": "Real -> Real -> Bool",
        "<": "Real -> Real -> Bool",
        ">": "Real -> Real -> Bool",
    }
)

LRA = DSL(__lra_syntax)


__nra_syntax = auto_type(
    {
        "+": "Real -> Real -> Real",
        "-": "Real -> Real -> Real",
        "*": "Real -> Real -> Real",
        "*": "Real -> Real -> Real",
        "/": "Real -> Real -> Real",
        "ite": "Bool -> Real -> Real -> Real",
        "=": "Real -> Real -> Bool",
        "<=": "Real -> Real -> Bool",
        ">=": "Real -> Real -> Bool",
        "<": "Real -> Real -> Bool",
        ">": "Real -> Real -> Bool",
    }
)

NRA = DSL(__nra_syntax)

__bv_syntax = auto_type(
    {
        "bvnot": "BitVector32 -> BitVector32",
        "bvand": "BitVector32 -> BitVector32 -> BitVector32",
        "bvor": "BitVector32 -> BitVector32 -> BitVector32",
        "bvneg": "BitVector32 -> BitVector32",
        "bvadd": "BitVector32 -> BitVector32 -> BitVector32",
        "bvmul": "BitVector32 -> BitVector32 -> BitVector32",
        "bvudiv": "BitVector32 -> BitVector32 -> BitVector32",
        "bvurem": "BitVector32 -> BitVector32 -> BitVector32",
        "bvshl": "BitVector32 -> BitVector32 -> BitVector32",
        "bvlshr": "BitVector32 -> BitVector32 -> BitVector32",
        "ite": "Bool -> BitVector32 -> BitVector32 -> BitVector32",
        "bvult": "BitVector32 -> BitVector32 -> Bool",
    }
)

BV = DSL(__bv_syntax)
