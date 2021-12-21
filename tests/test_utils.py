from synth.utils import make_deterministic_hash, to_partial_fun


def test_deterministic() -> None:
    make_deterministic_hash()
    assert hash(int) == hash(int)


def test_to_partial_fun() -> None:
    plus = to_partial_fun(lambda x, y: x + y, 2)
    for a in range(10):
        plusa = plus(a)
        for b in range(20):
            assert a + b == plusa(b)
