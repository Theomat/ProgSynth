from typing import Any, Optional, Tuple
from synth.syntax.grammars.cfg import CFGState, NoneType
from synth.syntax.grammars.det_grammar import DerivableProgram
from synth.syntax.program import Primitive
from synth.syntax.type_system import Type


def cfg_bigram_without_depth(
    ctx: Tuple[Type, Tuple[CFGState, NoneType]]
) -> Optional[Tuple[DerivableProgram, int]]:
    """
    Abstract away a CFG into tuples of (parent, no_arg).
    We lose depth information.
    """
    _, (state, __) = ctx
    ngram, ___ = state
    if len(ngram) > 0:
        return ngram.last()
    return None


def cfg_bigram_without_depth_and_equi_prim(
    ctx: Tuple[Type, Tuple[CFGState, NoneType]]
) -> Optional[Tuple[DerivableProgram, int]]:
    """
    Abstract away a CFG into tuples of (parent, no_arg) and merging together equivalent primitives.
    We lose depth information.
    """
    _, (state, __) = ctx
    ngram, ___ = state
    if len(ngram) > 0:
        (P, i) = ngram.last()
        if isinstance(P, Primitive) and "@" in P.primitive:
            return Primitive(P.primitive[: P.primitive.find("@")], P.type), i
        return P, i
    return None


def primitive_presence(*args: Any) -> None:
    """
    Abstract away a grammar into just the presence or absence of a primitive.
    """
    return None
