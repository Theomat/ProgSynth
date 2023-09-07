# [SyGuS](https://sygus.org/)

The aim here is to provide ready to use scripts that can work with SyGuS files.

## Setup

### Installing the parser

The ``parsing`` folder need to be filled with the ``sygus/src`` folder from <https://github.com/SyGuS-Org/tools> which contains the latest version of the parser.

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
