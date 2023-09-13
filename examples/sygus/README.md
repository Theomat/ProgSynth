# [SyGuS](https://sygus.org/)

The aim here is to provide ready to use scripts that can work with SyGuS files.

## Setup

### Installing the parser

The ``parsing`` folder need to be filled with the ``sygus/src`` folder from <https://github.com/SyGuS-Org/tools>.

The files are not included so you can get the latest version yourself otherwise you can just run the following commands assuming your current folder is ``ProgSynth/``:

```bash
git clone https://github.com/SyGuS-Org/tools.git
mv tools/sygus/src/* examples/sygus/parsing
yes | rm -r tools
```

### Installing the dependencies

The parsing requires the following additional dependency:

```
ply
```

## Scripts

### DSL Pruning

This enables the sharpening of grammar automatically in the SyGuS format.
The script will directly update the grammar in the specification file so you can use any solver.
It takes as input the original SyGuS specification file and a JSON sharpening file which is just a JSON file containing a list of strings such as:

```json
[
    "(+ ^+,0 ^0)",
    "(- _ ^0)",
    "(not ^not,and,or)",
    "(and ^and _)",
    "(or ^and,or ^and)",
    "(ite ^not _ _)",
    "(<=,>=,= ^-,+ _)"
]
```
