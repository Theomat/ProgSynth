#!/bin/env bash
# ===================================================================
# PARAMETERS
# ===================================================================
SEED=2
# size of training dataset
TRAIN_SIZE=2500
TEST_SIZE=500
BATCH_SIZE=16
EPOCHS=2
# timeout in seconds
TIMEOUT=60 
# ===================================================================
# CONSTANTS
# ===================================================================
EXPERIMENT_FOLDER="./deepcoder_experiment/"
DEEPCODER_DATASET="./deepcoder.pickle"
TEST_DATASET="$EXPERIMENT_FOLDER/test.pickle"
TRAIN_DATASET="$EXPERIMENT_FOLDER/train.pickle"
MODEL_RAW_FILE="$EXPERIMENT_FOLDER/model_raw.pt"
MODEL_PRUNED_FILE="$EXPERIMENT_FOLDER/model_pruned.pt"
# ===================================================================
# MAIN CODE
# ===================================================================
# Check deepcoder dataset exists
if [  ! -f "$DEEPCODER_DATASET" ]; then
    echo "$DEEPCODER_DATASET is missing!"
    exit
fi
# Check experiment folder exists and create it 
mkdir -p $EXPERIMENT_FOLDER
# Make test dataset
if [  ! -f "$TEST_DATASET" ]; then
    echo "[Generation] Creating the test dataset."
    python examples/pbe/dataset_generator.py --dsl deepcoder --dataset $DEEPCODER_DATASET --seed $SEED --size $TEST_SIZE -o $TEST_DATASET --uniform
    if [ $? != "0" ]; then
        exit 1
    fi
fi
# Make train dataset
if [  ! -f "$TRAIN_DATASET" ]; then
    echo "[Generation] Creating the train dataset."
    python examples/pbe/dataset_generator.py --dsl deepcoder --dataset $TEST_DATASET --seed $SEED --size $TRAIN_SIZE -o $TRAIN_DATASET --uniform
    if [ $? != "0" ]; then
        exit 1
    fi
fi
# Train raw model
if [  ! -f "$MODEL_RAW_FILE" ]; then
    echo "[Training] Creating the model from the raw DSL."
    python examples/pbe/model_trainer.py --dsl deepcoder.raw --dataset $TRAIN_DATASET --seed $SEED --b $BATCH_SIZE -o $MODEL_RAW_FILE -e $EPOCHS
    if [ $? != "0" ]; then
        exit 2
    fi
fi
# Train pruned model
if [  ! -f "$MODEL_PRUNED_FILE" ]; then
    echo "[Training] Creating the model from the pruned DSL."
    python examples/pbe/model_trainer.py --dsl deepcoder --dataset $TRAIN_DATASET --seed $SEED --b $BATCH_SIZE -o $MODEL_PRUNED_FILE -e $EPOCHS
    if [ $? != "0" ]; then
        exit 3
    fi
fi
# Train raw model
echo "[Evaluation] Evaluating the model from the raw DSL."
python examples/pbe/evaluate.py --dsl deepcoder.raw --dataset $TEST_DATASET --b $BATCH_SIZE --model $MODEL_RAW_FILE -o $EXPERIMENT_FOLDER -t $TIMEOUT
if [ $? != "0" ]; then
    exit 4
fi
# Train pruned model
echo "[Evaluation] Evaluating the model from the pruned DSL."
python examples/pbe/evaluate.py --dsl deepcoder --dataset $TEST_DATASET --b $BATCH_SIZE --model $MODEL_PRUNED_FILE -o $EXPERIMENT_FOLDER -t $TIMEOUT
if [ $? != "0" ]; then
    exit 5
fi
# Plotting
python examples/plot_results.py --dataset $TEST_DATASET --folder $EXPERIMENT_FOLDER