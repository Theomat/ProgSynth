from synth.generation.sampler import LexiconSampler, ListSampler, UnionSampler
from synth.syntax.type_system import (
    BOOL,
    INT,
    STRING,
    List,
    Type,
    UnknownType,
)


def test_lexicon_sampling() -> None:
    lexicon = list(range(100))
    sampler = LexiconSampler(lexicon)
    for _ in range(1000):
        x = sampler.sample(type=INT)
        assert x in lexicon


def test_list_sampling() -> None:
    lexicon = list(range(100))
    sampler = LexiconSampler(lexicon)
    for max_depth in [2, 3, 4]:
        a = ListSampler(sampler, [0.2] * 5, max_depth=max_depth, seed=0)
        my_type: Type = INT
        for _ in range(max_depth - 1):
            my_type = List(my_type)
        for _ in range(100):
            l = a.sample(type=my_type)
            el = l
            for _ in range(max_depth - 1):
                assert isinstance(
                    el, list
                ), f"Max depth:{max_depth} Type:{my_type} list:{l}"
                assert len(el) <= 5 and len(el) > 0
                el = l[0]
            b = a.sample(type=INT)
            assert isinstance(b, int) and b in lexicon


def test_union_sampler() -> None:
    lexicon = list(range(100))
    bool_lexicon = [True, False]
    str_lexicon = ["a", "b", "c", "d"]
    sampler = UnionSampler(
        {
            INT: LexiconSampler(lexicon),
            BOOL: LexiconSampler(bool_lexicon),
            STRING: LexiconSampler(str_lexicon),
        },
        LexiconSampler([[], []]),
    )

    for _ in range(100):
        x = sampler.sample(type=INT)
        assert isinstance(x, int) and x in lexicon
        y = sampler.sample(type=BOOL)
        assert isinstance(y, bool) and y in bool_lexicon
        z = sampler.sample(type=STRING)
        assert isinstance(z, str) and z in str_lexicon
        d = sampler.sample(type=UnknownType())
        assert isinstance(d, list) and len(d) == 0


def test_seeding() -> None:
    lexicon = list(range(100))
    aint = LexiconSampler(lexicon, seed=0)
    bint = LexiconSampler(lexicon, seed=0)
    for _ in range(1000):
        assert aint.sample(type=INT) == bint.sample(type=INT), _

    a = ListSampler(aint, [0.2] * 5, max_depth=3, seed=0)
    b = ListSampler(bint, [0.2] * 5, max_depth=3, seed=0)
    for _ in range(1000):
        assert a.sample(type=List(INT)) == b.sample(type=List(INT))
