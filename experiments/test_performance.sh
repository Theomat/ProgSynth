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
# timeout in seconds
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
    exit
fi
# Check experiment folder exists and create it 
mkdir -p $EXPERIMENT_FOLDER
# Make test dataset
if [  ! -f "$TEST_DATASET" ]; then
    echo "[Generation] Creating the test dataset."
    python examples/pbe/dataset_generator.py --dsl $DSL_NAME --dataset $TEST_DATASET --seed $SEED --size $TEST_SIZE -o $TEST_DATASET --uniform
    if [ $? != "0" ]; then
        exit 1
    fi
fi

# Make train dataset
if [  ! -f "$TRAIN_DATASET" ]; then
    echo "[Generation] Creating the train dataset."
    python examples/pbe/dataset_generator.py --dsl $DSL_NAME --dataset $TEST_DATASET --seed $SEED --size $TRAIN_SIZE -o $TRAIN_DATASET --uniform
    if [ $? != "0" ]; then
        exit 1
    fi
fi
# Train model
if [  ! -f "$MODEL_FILE" ]; then
    echo "[Training] Creating the model."
    python examples/pbe/model_trainer.py --dsl $DSL_NAME --dataset $TRAIN_DATASET --seed $SEED --b $BATCH_SIZE -o $MODEL_FILE -e $EPOCHS
    if [ $? != "0" ]; then
        exit 2
    fi
fi
# Eval model
echo "[Evaluation] Evaluating the model."
python examples/pbe/evaluate.py --dsl $DSL_NAME --dataset $TEST_DATASET --b $BATCH_SIZE --model $MODEL_FILE -o $EXPERIMENT_FOLDER -t $TIMEOUT
if [ $? != "0" ]; then
    exit 4
fi
# Plotting
python examples/pbe/plot_results.py --dataset $TEST_DATASET --folder $EXPERIMENT_FOLDER