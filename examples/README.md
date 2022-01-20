# Examples

This folder contains ready to use scripts and files that you can leverage to reproduce results from papers for example or to test your new ideas.

<!-- toc -->

- [Programming By Example](#programming-by-example)
  - [Deepcoder](#deepcoder)
    - [Downloading Deepcoder](#downloading-deepcoder)
  - [Dreamcoder](#dreamcoder)
    - [Downloading Dreamcoder](#downloading-dreamcoder)

<!-- tocstop -->

## Programming By Example

- The `pcfg_prediction.py` can load either the `deepcoder` or `dreamcoder` datasets, reproduces the task distribution, then train a neural net to predict the PCFG probabilities and compute the test loss on the dataset. Metrics are logged with [TensorBoard](https://www.tensorflow.org/tensorboard/) and a report of time spent is printed at the end of the script.
- The `dsl_analyser.py` can load either the `deepcoder` or `dreamcoder` datasets, reproduces the input distribution, then try to find all redundant derivations at depth 2 such as `(MAP[/4] (MAP[*4] var0))` or `(LENGTH (SORT var0))`. For examples removing these patterns in the deepcoder CFG, reduces its size by 10.5% at depth 5.

### DeepCoder

This is a dataset of:
> *M. Balog, A. L. Gaunt, M. Brockschmidt, S. Nowozin, and D. Tarlow.* Deep-coder: Learning to write programs. In International Conference on Learning Representations, ICLR, 2017. URL <https://openreview.net/forum?id=ByldLrqlx>.

This folder contains two files.
The `deepcoder.py` file contains an impelementation of the DSL along with a default evaluator.
The `convert_deepcoder.py` is a runnable python script which enables you to convert the original deepcoder dataset files to the ProgSynth format.

#### Downloading Deepcoder

You can download the archive from here: <https://storage.googleapis.com/deepcoder/dataset.tar.gz>. Then you simply need to:

```bash
gunzip dataset.tar.gz
tar -xf dataset.tar
```

You should see a few JSON files, these JSON files are now convertible with the `convert_deepcoder.py` script.

### Dreamcoder

This is the list dataset of:
> *K. Ellis, C. Wong, M. I. Nye, M. Sabl ́e-Meyer, L. Morales, L. B. Hewitt, L. Cary, A. Solar-Lezama, and J. B. Tenenbaum.* Dreamcoder: bootstrapping inductive program synthesis with wake-sleep library learning. In International Conference on Programming Language Design and Implementation, PLDI, 2021. URL <https://doi.org/10.1145/3453483.3454080>.

This folder contains two files.
The `dreamcoder.py` file contains an impelementation of the DSL along with a default evaluator.
The `convert_dreamcoder.py` is a runnable python script which enables you to convert the original dreamcoder dataset files to the ProgSynth format.

#### Downloading Dreamcoder

The dataset can be downloaded at <https://raw.githubusercontent.com/ellisk42/ec/master/data/list_tasks.json>.
This JSON file is now convertible with the `convert_dreamcoder.py` script.
