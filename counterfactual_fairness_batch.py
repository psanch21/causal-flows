import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default=None)

args = parser.parse_args()


folder_path = args.folder

# Iterate over folders within the specified folder
for folder_name in os.listdir(folder_path):
    folder = os.path.join(folder_path, folder_name)
    if os.path.isdir(folder):
        # Perform operations on each folder
        print(folder)
        os.system(f"python counterfactual_fairness.py --load_model {folder}")
