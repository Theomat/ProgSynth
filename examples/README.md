# Examples

This folder contains ready to use scripts and files that you can leverage to reproduce results from papers for example or to test your new ideas.

<!-- toc -->

- [Deepcoder](#deepcoder)
  - [Downloading Deepcoder](#downloading-deepcoder)
- [Dreamcoder](#dreamcoder)
  - [Downloading Dreamcoder](#downloading-dreamcoder)

<!-- tocstop -->

## DeepCoder

This is a dataset of:
> *M. Balog, A. L. Gaunt, M. Brockschmidt, S. Nowozin, and D. Tarlow.* Deep-coder: Learning to write programs. In International Conference on Learning Representations, ICLR, 2017. URL <https://openreview.net/forum?id=ByldLrqlx>.

This folder contains two files.
The `deepcoder.py` file contains an impelementation of the DSL along with a default evaluator.
The `convert_deepcoder.py` is a runnable python script which enables you to convert the original deepcoder dataset files to the ProgSynth format.

### Downloading Deepcoder

You can download the archive from here: <https://storage.googleapis.com/deepcoder/dataset.tar.gz>. Then you simply need to:

```bash
gunzip dataset.tar.gz
tar -xf dataset.tar
```

You should see a few JSON files, these JSON files are now convertible with the `convert_deepcoder.py` script.

## Dreamcoder

This is the list dataset of:
> *K. Ellis, C. Wong, M. I. Nye, M. Sabl ́e-Meyer, L. Morales, L. B. Hewitt, L. Cary, A. Solar-Lezama, and J. B. Tenenbaum.* Dreamcoder: bootstrapping inductive program synthesis with wake-sleep library learning. In International Conference on Programming Language Design and Implementation, PLDI, 2021. URL <https://doi.org/10.1145/3453483.3454080>.

This folder contains two files.
The `dreamcoder.py` file contains an impelementation of the DSL along with a default evaluator.
The `convert_dreamcoder.py` is a runnable python script which enables you to convert the original dreamcoder dataset files to the ProgSynth format.

### Downloading Dreamcoder

The dataset can be downloaded at <https://raw.githubusercontent.com/ellisk42/ec/master/data/list_tasks2.json>.
This JSON file is now convertible with the `convert_dreamcoder.py` script.
