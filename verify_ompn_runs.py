import subprocess
import os
import re

# Tolerance for floating point comparison
TOLERANCE = 1e-2

# Path to the OMPN folder (adjust if needed)
ompn_folder = os.path.join(os.path.dirname(__file__), "OMPN_Scripts")

# List all .sh files in the folder
sh_files = [f for f in os.listdir(ompn_folder) if f.endswith('.sh')]

for sh_file in sh_files:
    sh_path = os.path.join(ompn_folder, sh_file)
    print(f"Running {sh_file}...")
    try:
        result = subprocess.run(['bash', sh_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)

        output = result.stdout + result.stderr

        # Extract relevant values
        miou_full_match = re.search(r'test_miou_full\s+([0-9.]+)', output)
        miou_per_match = re.search(r'test_miou_per\s+([0-9.]+)', output)
        f1_full_match = re.search(r'test_f1_full\s+([0-9.]+)', output)
        f1_per_match = re.search(r'test_f1_per\s+([0-9.]+)', output)
        mof_full_match = re.search(r'test_mof_full\s+([0-9.]+)', output)
        mof_per_match = re.search(r'test_mof_per\s+([0-9.]+)', output)

        if miou_full_match and miou_per_match:
            miou_full = float(miou_full_match.group(1))
            miou_per = float(miou_per_match.group(1))
            f1_full = float(f1_full_match.group(1))
            f1_per = float(f1_per_match.group(1)) 
            mof_full = float(mof_full_match.group(1))
            mof_per = float(mof_per_match.group(1))

            print(f"Results for {sh_file}:")
            print(f"{f1_full},{f1_per},{miou_full},{miou_per},{mof_full},{mof_per}")
            print("=" * 40)
        else:
            print(f"[ERROR] {sh_file} — Failed to extract all required values")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {sh_file} — Script execution failed:\n{e.stderr}")
