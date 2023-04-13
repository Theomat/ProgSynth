# PBE Examples

This folder contains ready to use scripts and files that you can leverage to reproduce results from papers for example or to test your new ideas.

<!-- toc -->

- [Scripts](#scripts)
  - [Dataset Manipulation](#dataset-manipulation)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [DSL Manipulation](#dsl-manipulation)
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
## Scripts

### Dataset Manipulation

You can **explore** a dataset with `dataset_explorer.py`. It will provide you with an interactive prompt to explore the dataset. Use `help` to see the list of commands in the interactive prompt.

You can **generate synthetic datasets** based on a existing dataset, reproducing their distribution with:

- `dataset_generator.py`
- `dataset_generator_unique.py` which has the constraint that programs must uniquely identifiable from the examples. This is the **recommended way** of generating a dataset.

You can **improve solutions of a dataset** with `dataset_improve.py`. It takes a dataset and a solution file (obtained with `evaluate.py`) and replace the solutions of the dataset by the ones found if they are shorter.

### Model Training and Evaluation

You can **train a model** with `model_trainer.py`. It loads a dataset then train a neural net to predict the probabilities of the grammar. Metrics are logged with [TensorBoard](https://www.tensorflow.org/tensorboard/.

You can **evaluate a model** with `evaluate.py`. It loads a dataset, a model, and runs our synthesis algorithm on every task trying to find a correct solution to the task.

You can **plot the results** of `evaluate.py` with `plot_results.py` which is located in the parent folder.

### DSL Manipulation

You can **find equations automatically for a given DSL** with `dsl_equation_generator.py`. loads a dataset and tries to find all redundant derivations at depth 2 such as `(MAP[/4] (MAP[*4] var0))` or `(LENGTH (SORT var0))` and produces constraints to forbid them.

You can **learn new primitives** with `dataset_learner.py`. It loads a dataset and try to learn a new primitive that would most help with expressing the dataset.

## DSLs

Here is an exhaustive list of available DSLs with this specification.

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

This is a dataset of string manipulation, it contains per-task constants, no polymrphic types nor need lambdas.
This is a dataset in the idea of the string manipulation dataset of FlashFill:
> *S. Gulwani* Automating String Processing in Spreadsheets using Input-Output Examples. PoPL'11, January 26-28, 2011, Austin, Texas, USA. URL <https://www.microsoft.com/en-us/research/publication/automating-string-processing-spreadsheets-using-input-output-examples/>

#### Downloading SyGus

The SyGus dataset can be found at <https://github.com/ellisk42/ec/tree/master/data/sygus>.
The SL files were converted and compressed in some JSON file that is provided in the folder.

## Example Pipeline

Here is an example pipeline for a DSL.
This would first produce a train and test dataset, then train a model, evaluate it then plot the results.

```bash
#!/bin/env bash
# ===================================================================
# PARAMETERS
# ===================================================================
DSL_NAME="transduction"
BASE_DATASET="./flashfill.pickle"
SEED=2
### TASK PARAMETERS
# maximum depth of programs
MAX_DEPTH=5
# maximum number of examples per task
MAX_EXAMPLES=5
### MODEL TRAINING PARAMETERS
# size of training dataset
TRAIN_SIZE=2500
BATCH_SIZE=16
EPOCHS=2
### EVALUATION PARAMETERS
TEST_DATASET="$BASE_DATASET"
# useful only if test dataset != base dataset
TEST_SIZE=500 
# timeout in seconds for the evaluation
TIMEOUT=60 
# ===================================================================
# CONSTANTS
# ===================================================================
EXPERIMENT_FOLDER="./${DSL_NAME}_experiment/"
TRAIN_DATASET="$EXPERIMENT_FOLDER/train.pickle"
MODEL_FILE="$EXPERIMENT_FOLDER/model.pt"
# ===================================================================
# MAIN CODE
# ===================================================================
# Check base dataset exists
if [  ! -f "$BASE_DATASET" ]; then
    echo "$BASE_DATASET is missing!"
    exit 1
fi
# Check experiment folder exists and create it 
mkdir -p $EXPERIMENT_FOLDER
# Make test dataset
if [  ! -f "$TEST_DATASET" ]; then
    echo "[Generation] Creating the test dataset."
    python examples/pbe/dataset_generator_unique.py --dsl $DSL_NAME --dataset $TEST_DATASET --seed $SEED --programs $TEST_SIZE --inputs 2 -o $TEST_DATASET --max-depth $MAX_DEPTH --max-examples $MAX_EXAMPLES
    if [ $? != "0" ]; then
        exit 2
    fi
fi

# Make train dataset
if [  ! -f "$TRAIN_DATASET" ]; then
    echo "[Generation] Creating the train dataset."
    python examples/pbe/dataset_generator_unique.py --dsl $DSL_NAME --dataset $TEST_DATASET --seed $SEED --programs $TRAIN_SIZE -o $TRAIN_DATASET --inputs 2 --max-depth $MAX_DEPTH --max-examples $MAX_EXAMPLES
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
