#!/usr/bin/env bash

# Generate a random number (timestamp)
CURR_DATE_TIME=$(date +%s)

# Training configuration flags
COMPILE_TRAIN_STEPS=3000
COMPILE_EVAL_FREQ=10
COMPILE_LR=0.001
COMPILE_BATCH_SIZE=256
COMPILE_MAX_SEGS=4
COMPILE_SKILLS=2
COMPILE_BETA_Z=0.1
COMPILE_BETA_B=0.1
COMPILE_PRIOR_RATE=3

ENVS="wsws_static_symbolic"
STATE_SIZE=1087
ACTION_SIZE=16

# Model configuration
HIDDEN_SIZE=128

# Misc
DEBUG=false
MODE="compile"
EXPERIMENT="testing_gmm_wsws_static_symbolic_${CURR_DATE_TIME}"
CUDA=true
PROCS=8

# Run the Python training script with flags
python main.py \
  --mode "$MODE" \
  --cuda "$CUDA" \
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
  --debug "$DEBUG" \
  --experiment "$EXPERIMENT" \
  --procs "$PROCS"
