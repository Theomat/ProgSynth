import os


def make_deterministic_hash() -> None:
    """
    Make sure Python hash are deterministic.
    """
    hashseed = os.getenv("PYTHONHASHSEED")
    if not hashseed:
        os.environ["PYTHONHASHSEED"] = "0"
