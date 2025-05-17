#!/bin/bash

ENVS="wsws_static_pixels"
MAX_SEGS=4
SKILLS=2

python optuna_study.py --task "$ENVS" --max-segs "$MAX_SEGS" --skills "$SKILLS"
