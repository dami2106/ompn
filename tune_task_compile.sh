#!/bin/bash

BASE_TASK="stone_pick_static"
MAX_SEGS=8
SKILLS=5
TRIALS=150  # Set your desired number of trials here

# Generate task variants
TASK_VARIANTS=(
  "${BASE_TASK}_symbolic"
  # "${BASE_TASK}_symbolic_big"
  "${BASE_TASK}_pixels"
  # "${BASE_TASK}_pixels_big"
)

# Start timing
start_time=$(date +%s)

# Run the study for each variant
for TASK in "${TASK_VARIANTS[@]}"; do
  echo "Running study for task: $TASK"
  python optuna_study_compile.py --task "$TASK" --max-segs "$MAX_SEGS" --skills "$SKILLS" --trials "$TRIALS"
  echo "Study for task $TASK completed."
done

# End timing
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Total time for all studies: ${elapsed} seconds"

#Delete the experiment directory
rm -rf experiments/
