#!/usr/bin/env bash

#Generate a random number
CURR_DATE_TIME=$(date +%s)

python main.py --mode compile --compile_train_steps 7500 --arch omstack --nb_slots 3 --envs stone_pick_dataset --env_arch noenv --experiment static_pick_comparison_long_train_7500

# After compilation, remove the 'experiments' folder
rm -rf experiments

