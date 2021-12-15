from synth.utils import make_deterministic_hash


def test_deterministic() -> None:
    make_deterministic_hash()
    assert hash(int) == hash(int)
