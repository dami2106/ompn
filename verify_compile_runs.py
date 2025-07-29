import subprocess
import os
import re

# Tolerance for floating point comparison
TOLERANCE = 1e-2

# Path to the OMPN folder (adjust if needed)
ompn_folder = os.path.join(os.path.dirname(__file__), "CompILE_Scripts")

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
        expected_match = re.search(r'Expected\s+([0-9.]+)', output)

        if miou_full_match and miou_per_match and expected_match:
            miou_full = float(miou_full_match.group(1))
            miou_per = float(miou_per_match.group(1))
            expected = float(expected_match.group(1))

            weighted_miou = 0.8 * miou_full + 0.2 * miou_per

            if abs(weighted_miou - expected) > TOLERANCE:
                print(f"[FAIL] {sh_file} — Weighted mIoU ({weighted_miou:.4f}) != Expected ({expected:.4f})")
        else:
            print(f"[ERROR] {sh_file} — Failed to extract all required values")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {sh_file} — Script execution failed:\n{e.stderr}")
