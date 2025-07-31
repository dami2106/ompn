#!/usr/bin/env bash

# Generate a random number (timestamp)
CURR_DATE_TIME=$(date +%s)

# Training configuration flags
COMPILE_BATCH_SIZE=256
COMPILE_BETA_B=0.1
COMPILE_BETA_Z=0.01
COMPILE_LR=0.0001
COMPILE_PRIOR_RATE=10
COMPILE_TRAIN_STEPS=5000
HIDDEN_SIZE=256

ENVS="stone_pick_static_pixels_big"
EXPERIMENT="stone_pick_static_pixels_big"
COMPILE_MAX_SEGS=8
COMPILE_SKILLS=5

STATE_SIZE=650
ACTION_SIZE=16
COMPILE_EVAL_FREQ=10

# Misc
DEBUG=false
MODE="compile"
CUDA=true
PROCS=8

# Conditional flags
CUDA_FLAG=""
if [ "$CUDA" = "true" ]; then
  CUDA_FLAG="--cuda"
fi

DEBUG_FLAG=""
if [ "$DEBUG" = "true" ]; then
  DEBUG_FLAG="--debug"
fi

# Run the Python training script with flags
python main.py \
  --mode "$MODE" \
  $CUDA_FLAG \
  $DEBUG_FLAG \
  --compile_train_steps "$COMPILE_TRAIN_STEPS" \
  --compile_eval_freq "$COMPILE_EVAL_FREQ" \
  --compile_lr "$COMPILE_LR" \
  --compile_batch_size "$COMPILE_BATCH_SIZE" \
  --compile_max_segs "$COMPILE_MAX_SEGS" \
  --compile_skills "$COMPILE_SKILLS" \
  --compile_beta_z "$COMPILE_BETA_Z" \
  --compile_beta_b "$COMPILE_BETA_B" \
  --compile_prior_rate "$COMPILE_PRIOR_RATE" \
  --compile_state_size "$STATE_SIZE" \
  --compile_action_size "$ACTION_SIZE" \
  --envs "$ENVS" \
  --hidden_size "$HIDDEN_SIZE" \
  --experiment "$EXPERIMENT" \
  --procs "$PROCS"\
  --seed 10

echo "Training script done. Expected 0.27690"
