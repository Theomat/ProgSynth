#!/bin/env bash
# ===================================================================
# PARAMETERS
# ===================================================================
SEED=2
# size of training dataset
TRAIN_SIZE=2500
BATCH_SIZE=16
EPOCHS=2
# timeout in seconds
TIMEOUT=60 
# ===================================================================
# CONSTANTS
# ===================================================================
EXPERIMENT_FOLDER="./dreamcoder_experiment/"
DREAMCODER_DATASET="./dreamcoder.pickle"
TEST_DATASET=$DREAMCODER_DATASET
TRAIN_DATASET="$EXPERIMENT_FOLDER/train.pickle"
MODEL_CFG_FILE="$EXPERIMENT_FOLDER/model_cfg.pt"
MODEL_UCFG_FILE="$EXPERIMENT_FOLDER/model_ucfg.pt"
# ===================================================================
# MAIN CODE
# ===================================================================
# Check dreamcoder dataset exists
if [  ! -f "$DREAMCODER_DATASET" ]; then
    echo "$DREAMCODER_DATASET is missing!"
    exit
fi
# Check experiment folder exists and create it 
mkdir -p $EXPERIMENT_FOLDER
# Make train dataset
if [  ! -f "$TRAIN_DATASET" ]; then
    echo "[Generation] Creating the train dataset."
    python examples/pbe/dataset_generator.py --dsl dreamcoder --dataset $TEST_DATASET --seed $SEED --size $TRAIN_SIZE -o $TRAIN_DATASET --uniform --constrained
    if [ $? != "0" ]; then
        exit 1
    fi
fi
# Train cfg model
if [  ! -f "$MODEL_CFG_FILE" ]; then
    echo "[Training] Creating the CFG model."
    python examples/pbe/model_trainer.py --dsl dreamcoder --dataset $TRAIN_DATASET --seed $SEED --b $BATCH_SIZE -o $MODEL_CFG_FILE -e $EPOCHS
    if [ $? != "0" ]; then
        exit 2
    fi
fi
# Evaluate cfg model
echo "[Evaluation] Evaluating the model from the raw DSL."
python examples/pbe/evaluate.py --dsl dreamcoder --dataset $TEST_DATASET --b $BATCH_SIZE --model $MODEL_CFG_FILE -o $EXPERIMENT_FOLDER -t $TIMEOUT
if [ $? != "0" ]; then
    exit 3
fi
# Train cfg model
if [  ! -f "$MODEL_UCFG_FILE" ]; then
    echo "[Training] Creating the CFG model."
    python examples/pbe/model_trainer.py --dsl dreamcoder --dataset $TRAIN_DATASET --seed $SEED --b $BATCH_SIZE -o $MODEL_UCFG_FILE -e $EPOCHS --constrained
    if [ $? != "0" ]; then
        exit 4
    fi
fi
# Evaluate cfg model
echo "[Evaluation] Evaluating the model from the raw DSL."
python examples/pbe/evaluate.py --dsl dreamcoder --dataset $TEST_DATASET --b $BATCH_SIZE --model $MODEL_UCFG_FILE -o $EXPERIMENT_FOLDER -t $TIMEOUT --constrained
if [ $? != "0" ]; then
    exit 5
fi
# Plotting
python examples/plot_results.py --dataset $TEST_DATASET --folder $EXPERIMENT_FOLDER