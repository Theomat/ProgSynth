"""
This file provides useful structures in order to treat with regular expressions (regex).
Inspired by pregex project of github user Hieuzest: https://github.com/Hieuzest/pregex/blob/master/pregex.py
"""
from typing import Any, Callable, Dict, List, Tuple, Optional, Match
from abc import ABC, abstractmethod
from functools import lru_cache
import re
import enum

from synth.syntax.type_system import PrimitiveType


REGEXP = PrimitiveType("regexp")


def escape_regex(s: str):
    return re.escape(s)


def escape(s: str, *, escape_comma: bool = True) -> str:
    s = s.replace("&", "&amp;").replace("[", "&#91;").replace("]", "&#93;")
    if escape_comma:
        s = s.replace(",", "&#44;")
    return s


def unescape(s: str) -> str:
    return (
        s.replace("&#44;", ",")
        .replace("&#91;", "[")
        .replace("&#93;", "]")
        .replace("&amp;", "&")
    )


class RegexFlag(enum.IntFlag):
    # split character (whitespace)
    SPLIT: int = 2**21


class _RepeatState:
    def __init__(self) -> None:
        super().__init__()
        self._n = 0
        self._map: Dict[str, List[str]] = {}
        self._map[None] = []
        self._stack: List[str] = []
        self._patterns = {}

    @property
    def current(self) -> Optional[str]:
        return self._stack[-1] if self._stack else None

    def __enter__(self) -> "_RepeatState":
        self._n += 1
        self._stack.append(f"__repeat__{self._n}__")
        self._map[self.current] = []
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self._stack.pop()
        self._n -= 1

    def add(self, name: str) -> None:
        self._map[self.current] = name

    def check(self, pattern) -> bool:
        self._patterns[self.current] = pattern
        return bool(self._map[self.current])

    def generate(self) -> Dict[str, Tuple[str, str]]:
        res = {}
        for k, v in self._map.items():
            if not k or k in self._stack:
                continue
            for name in v:
                res[name] = (k, self._patterns[k])
        return res


class _StructState:
    def __init__(self) -> None:
        super().__init__()
        self._n = 0

    def __enter__(self) -> "_StructState":
        self._n += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._n -= 1

    def check(self) -> bool:
        return bool(self._n)


class _RequireState(dict):
    pass


class State:
    def __init__(self) -> None:
        super().__init__()
        self.repeat = _RepeatState()
        self.struct = _StructState()
        self.require = _RequireState()


class AbsPattern(ABC):
    @property
    @lru_cache()
    def re(self) -> str:
        return str(compile(self)._pattern)

    @abstractmethod
    def __compile__(self, flags: RegexFlag, state: State) -> "Raw":
        pass

    def __str__(self) -> str:
        return self.re


class Raw(AbsPattern):
    """
    raw string representation of a pattern.
    Example:
    "\s*" is a raw pattern indicating "any blankspace"
    """

    def __init__(self, pattern) -> None:
        super().__init__()
        self._pattern = pattern

    def __compile__(self, flags: RegexFlag, state: State) -> "Raw":
        return self

    def __str__(self) -> str:
        return self._pattern

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._pattern!r})"


class Pattern(AbsPattern):
    """
    Examples:
    ["a", "b"] -> "(ab)" (concat)
    ("a", "b") -> "(a|b)" (choice)
    """

    def __init__(self, pattern) -> None:
        super().__init__()
        self._pattern = pattern

    def __compile__(self, flags: RegexFlag, state: State) -> "Raw":
        return super().__compile__(flags, state)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._pattern!r})"


AnyPlain = Raw(r"[^,\[\]]*")
AnyElement = Raw(r"[^,\[\]]+|(\[[^\]]*\])")
AnyMessage = Raw(r"([^,\[\]]|(\[[^\]]*\]))*")
AnyBlank = Raw(r"\s*")

_REQUIRE_DEFAULT_KEY = "__default_require_post__"


class CompiledPattern(Raw):
    def __init__(self, pattern, **kwargs) -> None:
        super().__init__(str(pattern))
        self._kwargs = kwargs

    # researches in a string a raw pattern
    def search(self, s: str, flags: RegexFlag = 0) -> "Match":
        m = re.search(self._pattern, s, flags=flags)
        return Match(match=m, pattern=self, flags=flags) if m else None

    def match(self, s: str, flags: RegexFlag = 0) -> "Match":
        m = re.match(self._pattern, s, flags=flags)
        return Match(match=m, pattern=self, flags=flags) if m else None

    def fullmatch(self, s: str, flags: RegexFlag = 0) -> "Match":
        m = re.fullmatch(self._pattern, s, flags=flags)
        return Match(match=m, pattern=self, flags=flags) if m else None

    def findall(self, s: str, flags: RegexFlag = 0) -> "Match":
        m = re.findall(self._pattern, s, flags=flags)
        return Match(match=m, pattern=self, flags=flags) if m else None


class Struct(AbsPattern):
    """
    Structure of a pattern, tl;dr special regex char
    """

    def __init__(self, type: str, **kwargs) -> None:
        self._type = type
        self._kwargs = kwargs

    def __compile__(self, flags: RegexFlag, state: State) -> "Raw":
        with state.struct:
            r = rf"(\[{_compile(self._type, flags=flags, state=state)}(?:,[a-zA-Z0-9-_.]+=[^,\]]+)*"
            kwargs = sorted(self._kwargs.items())
            for key, value in kwargs:
                r += (
                    rf",{_compile(key, flags=flags, state=state)}={_compile(value, flags=flags, state=state)}"
                    r"(?:,[a-zA-Z0-9-_.]+=[^,\]]+)*"
                )
            r += r",?\])"
        return Raw(r)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._type!r}, **{self._kwargs!r})"


class Require(AbsPattern):
    """
    Required pattern
    """

    def __init__(self, pattern: AnyPlain, name: str, post=...) -> None:
        self._pattern = pattern
        self._name = name
        self._post = post

    def __construct__(self, p: Raw) -> "Raw":
        return Raw(rf"(?P<{self._name}>{p}")

    def __compile__(self, flags: RegexFlag, state: State) -> "Raw":
        state.repeat.add(self._name)
        if self._post is ...:
            self._post = state.require.get(_REQUIRE_DEFAULT_KEY)
        if isinstance(self._post, Tuple):
            if isinstance(self._post[state.struct.check()], Tuple):
                state.require[self._name] = (
                    (unescape, *self._post[state.struct.check()])
                    if state.struct.check()
                    else self._post[state.struct.check()]
                )
            else:
                state.require[self._name] = (
                    (unescape, self._post[state.struct.check()])
                    if state.struct.check()
                    else self._post[state.struct.check()]
                )
        else:
            state.require[self._name] = (
                (unescape, self._post) if state.struct.check() else self._post
            )
        return self.__construct__(_compile(self._pattern, flags=flags, state=state))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._pattern!r}, name={self._name!r})"


class Repeat(AbsPattern):
    """
    Several ways to repeat a pattern: *, ?, + and numbered times
    """

    Kleene = -1
    Optional = -2
    Plus = -3

    def __init__(self, pattern, times: int = -1) -> None:
        super().__init__()
        if isinstance(pattern, list) and not isinstance(pattern, str):
            self._pattern = pattern
        else:
            self._pattern = [pattern]
        self._times = times

    def __construct__(self, p: Raw) -> "Raw":
        if self._times == self.Kleene:
            return Raw(f"{p._pattern}*")
        elif self._times == self.Optional:
            return Raw(f"{p._pattern}?")
        elif self._times == self.Plus:
            return Raw(f"{p._pattern}+")
        else:
            return Raw(f"{p._pattern}{{{self._times}}}")

    def __compile__(self, flags: RegexFlag, *, state: State) -> "Raw":
        with state.repeat as repeat:
            p = _compile(self._pattern, flags=flags, state=state, raw=False)
            if repeat.check(p):
                p = self.__construct__(p)
                p = Require(p, name=repeat.current)
            else:
                p = self.__construct__(p)
        return _compile(p, flags=flags, state=state)

    def __repr__(self) -> str:
        if type(self) == Repeat:
            return f"{type(self).__name__}({self._pattern!r}, times={self._times})"
        else:
            return f"{type(self).__name__}({self._pattern!r})"


class Kleene(Repeat):
    def __init__(self, pattern) -> None:
        super().__init__(pattern, times=self.Kleene)


class Optional(Repeat):
    def __init__(self, pattern) -> None:
        super().__init__(pattern, times=self.Optional)


class Plus(Repeat):
    def __init__(self, pattern) -> None:
        super().__init__(pattern, times=self.Plus)


class Match:
    def __init__(
        self, match: Match, pattern: CompiledPattern, flags: RegexFlag
    ) -> None:
        super().__init__()
        self._match = match
        self._pattern = pattern
        self._flags = flags
        self._repeat_map: Dict[
            str, Tuple[str, CompiledPattern]
        ] = self._pattern._kwargs["_repeat_map"]
        self._require_post: Dict[str, Callable] = self._pattern._kwargs["_require_post"]

    @property
    def pattern(self) -> CompiledPattern:
        return self._pattern

    @property
    def re(self) -> str:
        return self._pattern._pattern

    @property
    def match(self) -> Match:
        return self._match

    @property
    def string(self) -> str:
        return self._match.string

    @property
    def pos(self) -> int:
        return self._match.pos

    @property
    def endpos(self) -> int:
        return self._match.endpos

    @lru_cache
    def span(self, key=None, base: int = None):
        if not base:
            base = self.pos
        if not key:
            return (base + self.pos, base + self.endpos)

        if self._repeat_map.get(key):
            repeat_key, repeat_pattern = self._repeat_map.get(key)

            outer_spans = self.span(repeat_key, base=base)
            if isinstance(outer_spans, Tuple):
                outer_spans = [outer_spans]

            res = []
            for outer_span in outer_spans:
                outer_string = self.string[slice(*outer_span)]
                for m in re.finditer(repeat_pattern.re, outer_string):
                    inner_match = repeat_pattern.match(
                        outer_string[slice(*m.span())], flags=self._flags
                    )
                    r = inner_match.span(key, base=outer_span[0] + m.start())

                    if isinstance(r, Tuple):
                        res.append(r)
                    else:
                        res.extend(r)
            return res
        else:
            r = self._match.span(key)
            return (base + r[0], base + r[1])

    def start(self, key=None):
        span = self.span(key)
        if isinstance(span, Tuple):
            return span[0]
        else:
            return list(sp[0] for sp in span)

    def end(self, key=None):
        span = self.span(key)
        if isinstance(span, Tuple):
            return span[1]
        else:
            return list(sp[1] for sp in span)

    def _group(self, key=None):
        span = self.span(key)
        if isinstance(span, Tuple):
            return self.string[slice(*span)]
        else:
            return list(self.string[slice(*sp)] for sp in span)

    @lru_cache
    def group(self, key=None):
        post = self._require_post.get(key)
        if post and not isinstance(post, Tuple):
            post = (post,)

        def _post(x):
            for _p in post:
                if _p:
                    x = _p(x)
            return x

        res = self._group(key)
        if post:
            if isinstance(res, str):
                return _post(res)
            else:
                return list(_post(r) for r in res)
        else:
            return res

    def __str__(self) -> str:
        return f"<pregex.Match of {self._match}>"

    def __repr__(self) -> str:
        return self.__str__()


"""
This method avoids backreference. 
Not vital, but avoids wasting memory (we don't use catched groups)
"""


def _enclose(s: str):
    # return f"(?:{s})"
    return s


def _escape(s: str, code=True, comma=True):
    if code:
        return escape_regex(escape(s, escape_comma=comma))
    else:
        return escape_regex(s)


def _compile(pattern, flags: RegexFlag, *, state: State, raw: bool = True) -> "Raw":
    if isinstance(pattern, AbsPattern):
        p = pattern.__compile__(flags=flags, state=state)
    elif isinstance(pattern, str):
        p = Raw(_enclose(_escape(pattern)))
    elif isinstance(pattern, tuple):
        p = Raw(
            _enclose(
                "|".join(str(_compile(p, flags=flags, state=state)) for p in pattern)
            )
        )
    elif isinstance(pattern, list):
        if flags & RegexFlag.SPLIT:
            p = Raw(
                _enclose(
                    str(AnyBlank)
                    + str(AnyBlank).join(
                        str(_compile(p, flags=flags, state=state)) for p in pattern
                    )
                    + str(AnyBlank)
                )
            )
        else:
            p = Raw(
                _enclose(
                    "".join(str(_compile(p, flags=flags, state=state)) for p in pattern)
                )
            )
    else:
        raise ValueError(f"{pattern!r} is not a valid pattern!")
    if not raw:
        p = CompiledPattern(
            p,
            _repeat_map=state.repeat.generate(),
            _require_post=state.require,
        )
    return p


def compile(
    pattern, flags: RegexFlag = 0, *, default_require_post=...
) -> CompiledPattern:
    state = State()
    if default_require_post is not ...:
        if not isinstance(default_require_post, Tuple):
            default_require_post = (default_require_post, default_require_post)
        state.require[_REQUIRE_DEFAULT_KEY] = default_require_post

    return _compile(pattern, flags=flags, state=state, raw=False)


def regex_match(pattern, message: str, flags: RegexFlag = 0, **kwargs) -> "Match":
    if not isinstance(pattern, CompiledPattern):
        pattern = compile(pattern, flags=flags, **kwargs)
    return pattern.match(message, flags=flags)


def regex_fullmatch(pattern, message: str, flags: RegexFlag = 0, **kwargs) -> "Match":
    if not isinstance(pattern, CompiledPattern):
        pattern = compile(pattern, flags=flags, **kwargs)
    return pattern.fullmatch(message, flags=flags)


def regex_search(pattern, message: str, flags: RegexFlag = 0, **kwargs) -> "Match":
    if not isinstance(pattern, CompiledPattern):
        pattern = compile(pattern, flags=flags, **kwargs)
    return pattern.search(message, flags=flags)


def regex_match_rawstring(pattern, string: str, flags: RegexFlag = 0) -> "Match":
    if not isinstance(pattern, CompiledPattern):
        pattern = compile(pattern, flags=flags, default_require_post=unescape)
    return pattern.match(escape(string), flags=flags)


def regex_findall(pattern, message: str, flags: RegexFlag = 0, **kwargs) -> "Match":
    if not isinstance(pattern, CompiledPattern):
        pattern = compile(pattern, flags=flags, **kwargs)
    print(pattern)
    return pattern.findall(message, flags=flags)
