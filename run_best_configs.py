import csv
import subprocess
import re
import sys
from pathlib import Path

# Path to the CSV file
CSV_PATH = Path("Compile_Tuning/wsws_static_symbolic_big/results_wsws_static_symbolic_big.csv")

# Default values from train_comp.sh
DEFAULTS = {
    'compile_train_steps': 5000,
    'compile_eval_freq': 10,
    'compile_lr': 0.001,
    'compile_batch_size': 128,
    'compile_max_segs': 4,
    'compile_skills': 2,
    'compile_beta_z': 0.01,
    'compile_beta_b': 0.1,
    'compile_prior_rate': 3,
    'compile_state_size': 1087,
    'compile_action_size': 16,
    'hidden_size': 128,
    'envs': 'wsws_static_symbolic_big',
    'mode': 'compile',
    'cuda': True,
    'procs': 8,
    'debug': False,
}

# Mapping from CSV columns to CLI flags
CSV_TO_FLAG = {
    'params_compile_batch_size': 'compile_batch_size',
    'params_compile_beta_b': 'compile_beta_b',
    'params_compile_beta_z': 'compile_beta_z',
    'params_compile_lr': 'compile_lr',
    'params_compile_prior_rate': 'compile_prior_rate',
    'params_compile_train_steps': 'compile_train_steps',
    'params_hidden_size': 'hidden_size',
}

# Only use parameters that are in the CSV or in train_comp.sh
SCRIPT_FLAGS = [
    'compile_train_steps', 'compile_eval_freq', 'compile_lr', 'compile_batch_size',
    'compile_max_segs', 'compile_skills', 'compile_beta_z', 'compile_beta_b',
    'compile_prior_rate', 'compile_state_size', 'compile_action_size',
    'envs', 'hidden_size', 'experiment', 'mode', 'procs'
]

# Boolean flags
BOOL_FLAGS = {
    'cuda': '--cuda',
    'debug': '--debug',
}

def parse_csv(csv_path):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    # Convert value to float for sorting
    for row in rows:
        row['value'] = float(row['value'])
    # Sort by value descending
    rows.sort(key=lambda r: r['value'], reverse=True)
    return rows

def build_command(row, defaults, experiment_name):
    cmd = [sys.executable, 'main.py']
    # Start with defaults
    params = defaults.copy()
    # Override with CSV values
    for csv_col, flag in CSV_TO_FLAG.items():
        if csv_col in row and row[csv_col] != '':
            params[flag] = type(defaults[flag])(row[csv_col])
    # Set experiment name
    params['experiment'] = experiment_name
    # Add only the allowed script flags
    for flag in SCRIPT_FLAGS:
        value = params[flag]
        cmd.extend([f'--{flag}', str(value)])
    # Add boolean flags
    for key, cli_flag in BOOL_FLAGS.items():
        if params.get(key, False):
            cmd.append(cli_flag)
    return cmd

def parse_output(output):
    match = re.search(r"test_miou_full\s+([\d.]+)\s+test_miou_per\s+([\d.]+)", output)
    if match:
        miou_full = float(match.group(1))
        miou_per = float(match.group(2))
        return 0.8 * miou_full + 0.2 * miou_per
    return None

def main():
    rows = parse_csv(CSV_PATH)
    if not rows:
        print("No rows found in CSV.")
        return
    best_value = rows[0]['value']
    threshold = best_value * 0.95
    print(f"Best value in CSV: {best_value:.5f}. Threshold for stopping: {threshold:.5f}")
    for i, row in enumerate(rows):
        experiment_name = f"wsws_static_symbolic_big_csv_{i}"
        print(f"\n=== Running config #{i} (CSV value: {row['value']:.5f}) ===")
        cmd = build_command(row, DEFAULTS, experiment_name)
        print('Command:', ' '.join(map(str, cmd)))
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
            output = result.stdout + '\n' + result.stderr
        except subprocess.CalledProcessError as e:
            print(f"Run failed: {e}")
            print(e.stdout)
            print(e.stderr)
            continue
        score = parse_output(output)
        if score is not None:
            print(f"Parsed score: {score:.5f}")
            if score >= threshold:
                print(f"Score {score:.5f} is within 5% of the best value. Stopping.")
                break
        else:
            print("Could not parse score from output.")

if __name__ == '__main__':
    main() 