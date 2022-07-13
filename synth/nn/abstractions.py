from typing import Optional, Tuple
from synth.syntax.grammars.cfg import CFGState, NoneType
from synth.syntax.grammars.det_grammar import DerivableProgram
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
