Usage of ProgSynth
==================

The ProgSynth framework utilizes Programming By Exemples (PBE) in order to construct programs that will resolve specific tasks, given by said examples.
The folder :code:`./examples/pbe` contains ready-to-use scripts and files that you can leverage to reproduce results from papers, or that you can modify to test your new ideas.

The tutorial on section :doc:`tutorial` is based on the example of tasks based on additions and substractions between integers or between float numbers, displaying step-by-step how to create a new DSL that can be used by the framework.

Programming by examples with lists
----------------------------------
Two different datasets, on top of the tutorial folder, are presented in :code:`./examples/pbe`, that you can use to test the program. The tasks represented by these examples are list-related.

Files used for PBE
~~~~~~~~~~~~~~~~~~
* The :code:`dataset_generator.py` can load the `deepcoder`, `dreamcoder` or `calculator` datasets, reproduces the task distribution, and generate a new synthetic dataset from scratch.
* The :code:`dataset_explorer.py` can load one of those DSL and any datset and will provide you with an interactive prompt to explore the dataset. Use `help` to see the list of commands in the interactive prompt.
* The :code:`evaluate.py` can load the `deepcoder`, `dreamcoder` or `calculator` DSL along with a given test dataset, a model, and runs heap search on every task trying to find a correct solution to the task.
* The :code:`pcfg_prediction.py` can load the `deepcoder`, `dreamcoder` or `calculator` DSL along with a given training dataset, then train a neural net to predict the PCFG probabilities. Metrics are logged with `TensorBoard <https://www.tensorflow.org/tensorboard/>`_ and a report of time spent is printed at the end of the script.
* The :code:`dsl_analyser.py` can load the `deepcoder`, `dreamcoder` or `calculator` datasets, reproduces the input distribution, then try to find all redundant derivations at depth 2 such as `(MAP[/4] (MAP[*4] var0))` or `(LENGTH (SORT var0))`. For examples removing these patterns in the deepcoder CFG, reduces its size by 10.5% at depth 5.

Deepcoder
---------
This is a dataset of::
    *M. Balog, A. L. Gaunt, M. Brockschmidt, S. Nowozin, and D. Tarlow.* Deep-coder: Learning to write programs. In International Conference on Learning Representations, ICLR, 2017. URL https://openreview.net/forum?id=ByldLrqlx.

This folder contains two files.
The :code:`deepcoder.py` file contains an implementation of the DSL along with a default evaluator.
The :code:`convert_deepcoder.py` is a runnable python script which enables you to convert the original deepcoder dataset files to the ProgSynth format.

Downloading Deepcoder
~~~~~~~~~~~~~~~~~~~~~
You can download the archive from here: https://storage.googleapis.com/deepcoder/dataset.tar.gz. Then you simply need to

.. code:: bash

    gunzip dataset.tar.gz
    tar -xf dataset.tar


You should see a few JSON files, these JSON files are now convertible with the :code:`convert_deepcoder.py` script.

Dreamcoder
----------
This is the list dataset of::
    *K. Ellis, C. Wong, M. I. Nye, M. Sabl ́e-Meyer, L. Morales, L. B. Hewitt, L. Cary, A. Solar-Lezama, and J. B. Tenenbaum.* Dreamcoder: bootstrapping inductive program synthesis with wake-sleep library learning. In International Conference on Programming Language Design and Implementation, PLDI, 2021. URL https://doi.org/10.1145/3453483.3454080.

This folder contains two files.
The :code:`dreamcoder.py` file contains an impelementation of the DSL along with a default evaluator.
The :code:`convert_dreamcoder.py` is a runnable python script which enables you to convert the original dreamcoder dataset files to the ProgSynth format.

Downloading Dreamcoder
~~~~~~~~~~~~~~~~~~~~~~

The dataset can be downloaded at https://raw.githubusercontent.com/ellisk42/ec/master/data/list_tasks.json.
This JSON file is now convertible with the :code:`convert_dreamcoder.py` script.
