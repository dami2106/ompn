#!/bin/bash

BASE_TASKS=("stone_pick_static")  # Add more base tasks as needed
MAX_SEGS=8
SKILLS=5
TRIALS=50  # Set your desired number of trials here

# Start timing
start_time=$(date +%s)

for BASE_TASK in "${BASE_TASKS[@]}"; do
  # Generate task variants for each base task
  TASK_VARIANTS=(
    # "${BASE_TASK}_symbolic"
    # "${BASE_TASK}_symbolic_big"
    # "${BASE_TASK}_pixels"
    "${BASE_TASK}_pixels_big"
  )

  for TASK in "${TASK_VARIANTS[@]}"; do
    echo "Running study for task: $TASK"
    python optuna_study_compile.py --task "$TASK" --max-segs "$MAX_SEGS" --skills "$SKILLS" --trials "$TRIALS"
    echo "Study for task $TASK completed."
  done
done

# End timing
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Total time for all studies: ${elapsed} seconds"
