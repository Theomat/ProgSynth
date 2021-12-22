from synth.utils import to_partial_fun, gen_take


def test_to_partial_fun() -> None:
    plus = to_partial_fun(lambda x, y: x + y, 2)
    for a in range(10):
        plusa = plus(a)
        for b in range(20):
            assert a + b == plusa(b)


def test_gen_take() -> None:
    g = (x for x in range(10000))
    for i in range(10):
        l = gen_take(g, 100)
        assert l == list(range(i * 100, (i + 1) * 100))
