import synth


def test_deterministic() -> None:
    assert hash(int) == hash(int)
