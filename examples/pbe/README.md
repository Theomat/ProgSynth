# PBE Examples

This folder contains ready to use scripts and files that you can leverage to reproduce results from papers for example or to test your new ideas.

<!-- toc -->

- [DSLs](#dsls)
  - [Calculator](#calculator)
  - [Deepcoder](#deepcoder)
    - [Downloading Deepcoder](#downloading-deepcoder)
  - [Dreamcoder](#dreamcoder)
    - [Downloading Dreamcoder](#downloading-dreamcoder)
  - [Regexp](#regexp)
  - [Transductions](#transductions)
    - [Downloading SyGus](#downloading-sygus)
- [Example pipeline](#example-pipeline)

<!-- tocstop -->
Here is an exhaustive list of available scripts, we recommend running them with -h to see the available options:

- The `dataset_generator.py` loads a dataset, reproduces the task distribution, and generate a new synthetic dataset from scratch.
- The `dataset_explorer.py` loads a dataset and will provide you with an interactive prompt to explore the dataset. Use `help` to see the list of commands in the interactive prompt.
- The `evaluate.py` loads a dataset, a model, and runs heap search on every task trying to find a correct solution to the task.
- The `plot_results.py` plot the results files created by ``evaluate.py``.
- The `model_trainer.py` loads a dataset then train a neural net to predict the probabilities of the grammar. Metrics are logged with [TensorBoard](https://www.tensorflow.org/tensorboard/) and a report of time spent is printed at the end of the script.
- The `dataset_improve.py` takes a dataset and a solution file (obtained with `evaluate.py`) and replace the solutions of the dataset by the ones found if they are shorter.
- The `dsl_analyser.py` loads a dataset and tries to find all redundant derivations at depth 2 such as `(MAP[/4] (MAP[*4] var0))` or `(LENGTH (SORT var0))` and produces constraints to forbid them using pruning.

## DSLs

Here is an exhaustive list of available DSLs wit hthis specification.

### Calculator

This is a toy DSL made for the tutorial, it contains addition and substraction, the constants allowed are 1, 2 and 3.

### DeepCoder

This is a dataset of:
> *M. Balog, A. L. Gaunt, M. Brockschmidt, S. Nowozin, and D. Tarlow.* Deep-coder: Learning to write programs. In International Conference on Learning Representations, ICLR, 2017. URL <https://openreview.net/forum?id=ByldLrqlx>.

This folder contains two files.
The `deepcoder.py` file contains an implementation of the DSL along with a default evaluator.
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

## Example Pipeline

Here is an example pipeline for a DSL.
This would first produce a ttrain and test dataset, then train a model, evaluate it then plot the results.

```bash
#!/bin/env bash
# ===================================================================
# PARAMETERS
# ===================================================================
SEED=2
# size of training dataset
TRAIN_SIZE=2500
# useful only if test dataset != base dataset
TEST_SIZE=500 
BATCH_SIZE=16
EPOCHS=2
# timeout in seconds for the evaluation
TIMEOUT=60 

DSL_NAME="transduction"
BASE_DATASET="./flashfill.pickle"
TEST_DATASET="$BASE_DATASET"
# ===================================================================
# CONSTANTS
# ===================================================================
EXPERIMENT_FOLDER="./${DSL_NAME}_experiment/"
TRAIN_DATASET="$EXPERIMENT_FOLDER/train.pickle"
MODEL_FILE="$EXPERIMENT_FOLDER/model.pt"
# ===================================================================
# MAIN CODE
# ===================================================================
# Check deepcoder dataset exists
if [  ! -f "$BASE_DATASET" ]; then
    echo "$BASE_DATASET is missing!"
    exit 1
fi
# Check experiment folder exists and create it 
mkdir -p $EXPERIMENT_FOLDER
# Make test dataset
if [  ! -f "$TEST_DATASET" ]; then
    echo "[Generation] Creating the test dataset."
    python examples/pbe/dataset_generator.py --dsl $DSL_NAME --dataset $TEST_DATASET --seed $SEED --size $TEST_SIZE -o $TEST_DATASET
    if [ $? != "0" ]; then
        exit 2
    fi
fi

# Make train dataset
if [  ! -f "$TRAIN_DATASET" ]; then
    echo "[Generation] Creating the train dataset."
    python examples/pbe/dataset_generator.py --dsl $DSL_NAME --dataset $TEST_DATASET --seed $SEED --size $TRAIN_SIZE -o $TRAIN_DATASET
    if [ $? != "0" ]; then
        exit 3
    fi
fi
# Train model
if [  ! -f "$MODEL_FILE" ]; then
    echo "[Training] Creating the model."
    python examples/pbe/model_trainer.py --dsl DSL_NAME --dataset $TRAIN_DATASET --seed $SEED --b $BATCH_SIZE -o $MODEL_FILE -e $EPOCHS
    if [ $? != "0" ]; then
        exit 4
    fi
fi
# Eval model
echo "[Evaluation] Evaluating the model."
python examples/pbe/evaluate.py --dsl DSL_NAME --dataset $TEST_DATASET --b $BATCH_SIZE --model $MODEL_FILE -o $EXPERIMENT_FOLDER -t $TIMEOUT
if [ $? != "0" ]; then
    exit 5
fi
# Plotting
python examples/pbe/plot_results.py --dataset $TEST_DATASET --folder $EXPERIMENT_FOLDER
```