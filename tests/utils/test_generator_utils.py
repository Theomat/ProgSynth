from synth.utils.generator_utils import gen_take


def test_gen_take() -> None:
    g = (x for x in range(10000))
    for i in range(10):
        l = gen_take(g, 100)
        assert l == list(range(i * 100, (i + 1) * 100))
