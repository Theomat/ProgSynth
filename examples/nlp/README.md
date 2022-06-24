# Examples

This folder contains ready to use scripts and files that you can leverage to reproduce results from papers for example or to test your new ideas.

<!-- toc -->

- [Programming from Natural Language](#programming-by-example)
  - [CoNaLa](#conala)
    - [Downloading CoNaLa](#downloading-conala)

<!-- tocstop -->

## Programming from Natural Language

TODO

### CoNaLa

This is a dataset of:
> *Yin, Pengcheng and Deng, Bowen and Chen, Edgar and Vasilescu, Bogdan and Neubig, Graham.* Learning to Mine Aligned Code and Natural Language Pairs from Stack Overflow. In International Conference on Mining Software Repositories, MSR, 2018. URL <https://doi.org/10.1145/3196398.3196408>.

This folder contains two files.
The `conala.py` file contains an impelementation of the DSL along with a default evaluator.
The `convert_conala.py` is a runnable python script which enables you to convert the original CoNaLa dataset files to the ProgSynth format.

#### Downloading CoNaLa

You can download the archive from thei website: <https://conala-corpus.github.io/>. Then you simply need to:

```bash
unzip conala-corpus-v1.1.zip
```

You should see a few JSON files in a ``conala-corpus`` folder, these JSON files are now convertible with the `convert_conala.py` script.