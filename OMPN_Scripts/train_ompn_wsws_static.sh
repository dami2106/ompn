#!/usr/bin/env bash

### MODE & EXPERIMENT ###
MODE="IL"
EXPERIMENT="wsws_static_pixels_big"
ENVS="wsws_static_pixels_big"
ENV_ARCH="noenv"

### MODEL ###
ARCH="omstack"
NB_SLOTS=2
HIDDEN_SIZE=256

### IL TRAINING ###
IL_TRAIN_STEPS=1000
IL_DEMO_FROM_MODEL=False
IL_EVAL_FREQ=20
IL_SAVE_FREQ=200
IL_NO_DONE=False
IL_VAL_RATIO=0.05
IL_BATCH_SIZE=64

### OPTIMIZATION ###
IL_RECURRENCE=40
IL_LR=0.0001
IL_CLIP=0.8

### SYSTEM ###
CUDA=True
DEBUG=False
MINECRAFT=False

### CONDITIONAL BOOLEAN FLAGS ###
DEMO_FLAG=""
if [ "$IL_DEMO_FROM_MODEL" = "True" ]; then
  DEMO_FLAG="--il_demo_from_model"
fi

NO_DONE_FLAG=""
if [ "$IL_NO_DONE" = "True" ]; then
  NO_DONE_FLAG="--il_no_done"
fi

CUDA_FLAG=""
if [ "$CUDA" = "True" ]; then
  CUDA_FLAG="--cuda"
fi

DEBUG_FLAG=""
if [ "$DEBUG" = "True" ]; then
  DEBUG_FLAG="--debug"
fi

MINECRAFT_FLAG=""
if [ "$MINECRAFT" = "True" ]; then
  MINECRAFT_FLAG="--minecraft"
fi

### RUN ###
python main.py \
  --mode "$MODE" \
  $DEBUG_FLAG \
  --experiment "$EXPERIMENT" \
  --envs "$ENVS" \
  --env_arch "$ENV_ARCH" \
  --arch "$ARCH" \
  --nb_slots "$NB_SLOTS" \
  --hidden_size "$HIDDEN_SIZE" \
  $DEMO_FLAG \
  --il_train_steps "$IL_TRAIN_STEPS" \
  --il_eval_freq "$IL_EVAL_FREQ" \
  --il_save_freq "$IL_SAVE_FREQ" \
  $NO_DONE_FLAG \
  --il_val_ratio "$IL_VAL_RATIO" \
  --il_batch_size "$IL_BATCH_SIZE" \
  --il_recurrence "$IL_RECURRENCE" \
  --il_lr "$IL_LR" \
  --il_clip "$IL_CLIP" \
  $CUDA_FLAG \
  $MINECRAFT_FLAG \
  --seed 12

echo "Training script done. Expected 0.63136"
