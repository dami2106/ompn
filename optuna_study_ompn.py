import optuna
import subprocess
import json
import re
import pandas as pd
import joblib
import os
import argparse
from subprocess import PIPE
import random
import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

def save_study_progress(study, save_dir, task):
    """Saves the study results to a CSV file and the study object to a pickle file."""
    df = study.trials_dataframe()
    df.to_csv(os.path.join(save_dir, f'results_{task}.csv'), index=False)
    joblib.dump(study, os.path.join(save_dir, f'optuna_model_{task}.pkl'))

def objective(trial, args):
    """Objective function to optimize IL hyperparameters"""

    # Tunable hyperparameters
    params = {
        "il_train_steps": trial.suggest_categorical("il_train_steps", [250, 500, 1000, 3000, 5000]),
        "il_lr": trial.suggest_categorical("il_lr", [1e-5, 1e-4, 1e-3, 1e-1]),
        "il_batch_size": trial.suggest_categorical("il_batch_size", [64, 128]),
        "il_clip": trial.suggest_categorical("il_clip", [0.1, 0.2, 0.3, 0.8]),
        "il_recurrence": trial.suggest_categorical("il_recurrence", [10, 20, 30, 40]),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "nb_slots": trial.suggest_categorical("nb_slots", [2, 3, 4]),
    }

    # Fixed parameters
    fixed = {
        "mode": "IL",
        "experiment": "full_ompn_original_3",
        "envs": args.task,
        "env_arch": "noenv",
        "arch": "omstack",
        "il_demo_from_model": False,
        "il_eval_freq": 20,
        "il_save_freq": 200,
        "il_no_done": False,
        "il_val_ratio": 0.05,
        "cuda": True,
        "debug": True,
    }

    # Build CLI command
    cmd = ["python", "main.py"]
    for k, v in {**params, **fixed}.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])

    # Run subprocess and parse output
    try:
        result = subprocess.run(
            cmd,
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True,
            check=True
        )
        output = result.stdout

        match = re.search(r"test_miou_full\s+([\d.]+)\s+test_miou_per\s+([\d.]+)", output)
        if match:
            miou_full = float(match.group(1))
            miou_per = float(match.group(2))
            return (0.8 * miou_full) + (0.2 * miou_per)
        else:
            return float('-inf')

    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"stdout:\n{e.stdout}")
        print(f"stderr:\n{e.stderr}")
        return float('-inf')

   

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Path of the dataset to use')
    parser.add_argument('--trials',  default=500, type=int, help='nb of trials')
    args = parser.parse_args()

    # args.task = args.envs.split(".")[0]

    #Open the JSON file from dataset/data_stats.json
    with open(os.path.join('dataset', "data_stats.json"), 'r') as f:
        data_stats = json.load(f)
    
    space_sizes = data_stats['space_sizes']

    if 'pixels' in args.task:
        if 'big' in args.task:
            args.state_size = space_sizes['pixels_state_size_big']
        else:
            args.state_size = space_sizes['pixels_state_size']
    else:
        args.state_size = space_sizes['symbolic_state_size']

    args.action_size = space_sizes['action_space_size']

    directory = os.path.join('Tuning_OMPN', args.task)
    os.makedirs(directory, exist_ok=True)

    study_file = os.path.join(directory, f'optuna_model_{args.task}.pkl')
    if os.path.exists(study_file):
        print("Study file found. Resuming the study.")
        study = joblib.load(os.path.join(directory, f'optuna_model_{args.task}.pkl'))
    else:
        print("No study file found. Starting a new study.")
        study = optuna.create_study(direction="maximize")

    #Print all the info from above 
    print(f"Task: {args.task}")
    # print(f"State size: {args.state_size}")
    # print(f"Action size: {args.action_size}")
    # print(f"Max segments: {args.max_segs}")
    # print(f"Skills: {args.skills}")

    # Set random seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.trials,
        callbacks=[lambda study, trial: save_study_progress(study, directory, args.task)]
    )

    save_study_progress(study, directory, args.task)

    # Save best trial to best.csv
    best_trial = study.best_trial
    df_all = study.trials_dataframe()
    df_best = df_all[df_all["number"] == best_trial.number]
    df_best.to_csv(os.path.join(directory, f'best_{args.task}.csv'), index=False)