import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-root", "--root", type=str, default=None, help="Root folder")
parser.add_argument("-clean", "--clean", action="store_true")
parser.add_argument("-extensions", "--extensions", type=str, default=".txt+config.yaml")

args = parser.parse_args()

ext_list = []

for e in args.extensions.split("+"):
    ext_list.append(f"'*{e}'")

exts = " ".join(ext_list)
command = f"zip -r {args.root}.zip {args.root} -i {exts}"
print(f"{command}")
os.system(command)

if args.clean:
    raise NotImplementedError
