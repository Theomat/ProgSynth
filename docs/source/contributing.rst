Contributing to ProgSynth
=========================

Feel free to open an issue or pull request if you have any questions or suggestions.
If you plan to work on an issue, let us know in the issue thread so we can avoid duplicate work.

Before attempting to push, make sure that the following holds:

- No code formatting error, see [Code Formatting](#code-formatting);
- No type error, see [Typing](#typing);
- All tests pass, see [Testing](#testing).

Dev Setup
---------

These development dependencies can be be found in [pyproject.toml](./pyproject.toml).
If you are using a development install, they should be installed by default. Otherwise you need to install them manually.

Code Formatting
----------------

We use `Black <https://black.readthedocs.io/en/stable/>`_ for code formatting. Th exact  version used can be found in our `pyproject.toml`.
You can run the following to format all files:

```bash
black .
```

Typing
-------

We use `mypy <http://mypy-lang.org/>`_ to check typing. We require you to use type hints at all times. That means for all function signatures and all places where `mypy` can't deduce the full type, type hints should be placed.
You can check if there are no typing errors with:

```bash
mypy synth
```

Testing
--------

We use `Pytest <https://docs.pytest.org/en/latest/>`_.
Please ensure a few things:

- When adding a new feature, also add relevant tests.
- Tests should be deterministic. If your test depends on randomness, do not forget to seed.
- No test should fail when you commit.

Finally, you can run the tests with:

```bash
pytest .
```
