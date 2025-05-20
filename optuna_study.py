import optuna
import subprocess
import json
import re
import pandas as pd
import joblib
import os
import argparse
from subprocess import PIPE

def save_study_progress(study, save_dir, task):
    """Saves the study results to a CSV file and the study object to a pickle file."""
    df = study.trials_dataframe()
    df.to_csv(os.path.join(save_dir, f'results_{task}.csv'), index=False)
    joblib.dump(study, os.path.join(save_dir, f'optuna_model_{task}.pkl'))

def objective(trial, args):
    """Objective function to optimize hyperparameters"""
    # Define hyperparameter search space
    params = {
        "compile_train_steps": trial.suggest_categorical("compile_train_steps", [100, 500, 1000, 2500, 5000]),
        "compile_lr": trial.suggest_categorical("compile_lr", [ 1e-4, 1e-3, 1e-2, 1e-1]),
        "compile_batch_size": trial.suggest_categorical("compile_batch_size", [64, 128, 256]),
        "compile_beta_z": trial.suggest_categorical("compile_beta_z", [0.01, 0.1, 0.5, 1.0]),
        "compile_beta_b": trial.suggest_categorical("compile_beta_b",[0.01, 0.1, 0.5, 1.0]),
        "compile_prior_rate": trial.suggest_categorical("compile_prior_rate",[3, 10, 15]),
        "hidden_size": trial.suggest_categorical("hidden_size", [ 64, 128, 256, 512]),
    }


    fixed = {
        "mode": "compile",
        "cuda": True,
        "compile_state_size": args.state_size,
        "compile_action_size": args.action_size,
        "envs": args.task,
        "debug": True,
        "procs": 8,
        "compile_max_segs": args.max_segs,
        "compile_skills": args.skills,
        "compile_eval_freq": 250,
    }


        # Build CLI command
    cmd = ["python", "main.py"]
    for k, v in {**params, **fixed}.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])


    try:
        result = subprocess.run(
            cmd,
            stdout=PIPE,       # capture stdout
            stderr=PIPE,       # capture stderr
            universal_newlines=True,  # decode bytes to str
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

    """
    python optuna_study.py --dataset wsws_static/wsws_static_symbolic
      --directory Traces/wsws_static/wsws_static_symbolic --feature-name symbolic_obs --clusters 2 --layers "1087 512 40"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Path of the dataset to use')
    parser.add_argument('--trials',  default=500, type=int, help='nb of trials')
    parser.add_argument('--max-segs', default=10, type=int, help='max number of segments')
    parser.add_argument('--skills', default=5, type=int, help='number of skills')
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

    directory = os.path.join('Tuning', args.task)
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
    print(f"State size: {args.state_size}")
    print(f"Action size: {args.action_size}")
    print(f"Max segments: {args.max_segs}")
    print(f"Skills: {args.skills}")


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