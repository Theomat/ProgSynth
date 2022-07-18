#!/bin/env bash
# ===================================================================
# PARAMETERS
# ===================================================================
SEED=2
# size of training dataset
TRAIN_SIZE=1000
BATCH_SIZE=16
EPOCHS=2
# timeout in seconds
TIMEOUT=60 
# ===================================================================
# CONSTANTS
# ===================================================================
DEEPCODER_DATASET="./deepcoder.pickle"
EXPERIMENT_FOLDER="./deepcoder_experiment/"
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
# Make train dataset
if [  ! -f "$TRAIN_DATASET" ]; then
    echo "[Generation] Creating the train dataset."
    python examples/pbe/dataset_generator.py --dsl deepcoder --dataset $DEEPCODER_DATASET --seed $SEED --size $TRAIN_SIZE
fi
# Train raw model
if [  ! -f "$MODEL_RAW_FILE" ]; then
    echo "[Training] Creating the model from the raw DSL."
    python examples/pbe/pcfg_prediction.py --dsl deepcoder.raw --dataset $TRAIN_DATASET --seed $SEED --b $BATCH_SIZE -o $MODEL_RAW_FILE -e $EPOCHS
fi
# Train pruned model
if [  ! -f "$MODEL_PRUNED_FILE" ]; then
    echo "[Training] Creating the model from the pruned DSL."
    python examples/pbe/pcfg_prediction.py --dsl deepcoder.pruned --dataset $TRAIN_DATASET --seed $SEED --b $BATCH_SIZE -o $MODEL_PRUNED_FILE -e $EPOCHS
fi
# Train raw model
echo "[Evaluation] Evaluating the model from the raw DSL."
python examples/pbe/evaluate.py --dsl deepcoder.raw --dataset $DEEPCODER_DATASET --b $BATCH_SIZE --model $MODEL_RAW_FILE -o $EXPERIMENT_FOLDER -t $TIMEOUT
# Train pruned model
echo "[Evaluation] Evaluating the model from the pruned DSL."
python examples/pbe/evaluate.py --dsl deepcoder.pruned --dataset $DEEPCODER_DATASET --b $BATCH_SIZE --model $MODEL_PRUNED_FILE -o $EXPERIMENT_FOLDER -t $TIMEOUT

# Plotting