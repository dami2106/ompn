#!/usr/bin/env bash

### MODE & EXPERIMENT ###
MODE="IL"
EXPERIMENT=None
ENVS="wsws_static_symbolic"
ENV_ARCH="noenv"

### MODEL ###
ARCH="omstack"
NB_SLOTS=3
HIDDEN_SIZE=128

### IL TRAINING ###
IL_DEMO_FROM_MODEL=False
IL_TRAIN_STEPS=50
IL_EVAL_FREQ=20
IL_SAVE_FREQ=200
IL_NO_DONE=False
IL_VAL_RATIO=0.05
IL_BATCH_SIZE=128

### OPTIMIZATION ###
IL_RECURRENCE=30
IL_LR=0.0005
IL_CLIP=0.2

### SYSTEM ###
CUDA=True

### RUN ###
python main.py \
  --mode "$MODE" \
  --experiment "$EXPERIMENT" \
  --debug "$DEBUG" \
  --envs "$ENVS" \
  --env_arch "$ENV_ARCH" \
  --arch "$ARCH" \
  --nb_slots "$NB_SLOTS" \
  --hidden_size "$HIDDEN_SIZE" \
  --il_demo_from_model "$IL_DEMO_FROM_MODEL" \
  --il_train_steps "$IL_TRAIN_STEPS" \
  --il_eval_freq "$IL_EVAL_FREQ" \
  --il_save_freq "$IL_SAVE_FREQ" \
  --il_no_done "$IL_NO_DONE" \
  --il_val_ratio "$IL_VAL_RATIO" \
  --il_batch_size "$IL_BATCH_SIZE" \
  --il_recurrence "$IL_RECURRENCE" \
  --il_lr "$IL_LR" \
  --il_clip "$IL_CLIP" \
  --cuda "$CUDA" \
