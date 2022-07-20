# PBE Examples

This folder contains ready to use scripts and files that you can leverage to reproduce results from papers for example or to test your new ideas.

<!-- toc -->

- [Programming By Example](#programming-by-example)
  - [Calculator](#calculator)
  - [Deepcoder](#deepcoder)
    - [Downloading Deepcoder](#downloading-deepcoder)
  - [Dreamcoder](#dreamcoder)
    - [Downloading Dreamcoder](#downloading-dreamcoder)
  - [Regexp](#regexp)
  - [Transductions](#transductions)
    - [Downloading SyGus](#downloading-sygus)

<!-- tocstop -->


- The `dataset_generator.py` loads a dataset, reproduces the task distribution, and generate a new synthetic dataset from scratch.
- The `dataset_explorer.py` loads a dataset and will provide you with an interactive prompt to explore the dataset. Use `help` to see the list of commands in the interactive prompt.
- The `evaluate.py` loads a dataset, a model, and runs heap search on every task trying to find a correct solution to the task.
- The `plot_results.py` plot the results files created by ``evaluate.py``
- The `model_trainer.py` loads a dataset then train a neural net to predict the probabilities of the grammar. Metrics are logged with [TensorBoard](https://www.tensorflow.org/tensorboard/) and a report of time spent is printed at the end of the script.
- The `dataset_improve.py` takes a dataset and a solution file (obtained with `evaluate.py`) and replace the solutions of the dataset by the ones found if they are shorter.
- The `dsl_analyser.py` can load either the `deepcoder` or `dreamcoder` datasets, reproduces the input distribution, then try to find all redundant derivations at depth 2 such as `(MAP[/4] (MAP[*4] var0))` or `(LENGTH (SORT var0))`.

### Calculator

This is a toy DSL made for the tutorial, it contains addition and substraction, the constants allowed are 1, 2 and 3.

### DeepCoder

This is a dataset of:
> *M. Balog, A. L. Gaunt, M. Brockschmidt, S. Nowozin, and D. Tarlow.* Deep-coder: Learning to write programs. In International Conference on Learning Representations, ICLR, 2017. URL <https://openreview.net/forum?id=ByldLrqlx>.

This folder contains two files.
The `deepcoder.py` file contains an impelementation of the DSL along with a default evaluator.
The `convert_deepcoder.py` is a runnable python script which enables you to convert the original deepcoder dataset files to the ProgSynth format.

#### Downloading Deepcoder

This is an integer list manipulation dataset, it is easy to work with, it does not contain constants nor need lambdas.
You can download the archive from here: <https://storage.googleapis.com/deepcoder/dataset.tar.gz>. Then you simply need to:

```bash
gunzip dataset.tar.gz
tar -xf dataset.tar
```

You should see a few JSON files, these JSON files are now convertible with the `convert_deepcoder.py` script.

### Dreamcoder

This is an integer list manipulation dataset, it contains constants, polymorphic types and lambdas.
This is the list dataset of:
> *K. Ellis, C. Wong, M. I. Nye, M. Sabl ́e-Meyer, L. Morales, L. B. Hewitt, L. Cary, A. Solar-Lezama, and J. B. Tenenbaum.* Dreamcoder: bootstrapping inductive program synthesis with wake-sleep library learning. In International Conference on Programming Language Design and Implementation, PLDI, 2021. URL <https://doi.org/10.1145/3453483.3454080>.

This folder contains two files.
The `dreamcoder.py` file contains an implementation of the DSL along with a default evaluator.
The `convert_dreamcoder.py` is a runnable python script which enables you to convert the original dreamcoder dataset files to the ProgSynth format.

#### Downloading Dreamcoder

The dataset can be downloaded at <https://raw.githubusercontent.com/ellisk42/ec/master/data/list_tasks.json>.
This JSON file is now convertible with the `convert_dreamcoder.py` script.

### Regexp

This is a dataset where there are positive and negative examples the goal is to find a program representing a regular expression matching the positive examples and not the negative examples.

### Transductions

THis is a dataset of string manipulation, it contains per-task constants, no polymrphic types nor need lambdas.
This is a dataset in the idea of the string manipulation dataset of FlashFill:
> *S. Gulwani* Automating String Processing in Spreadsheets using Input-Output Examples. PoPL'11, January 26-28, 2011, Austin, Texas, USA. URL <https://www.microsoft.com/en-us/research/publication/automating-string-processing-spreadsheets-using-input-output-examples/>

#### Downloading SyGus

The SyGus dataset can be found at <https://github.com/ellisk42/ec/tree/master/data/sygus>.
The SL files were converted and compressed in some JSON file that is provided in the folder.
