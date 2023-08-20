import argparse

from causal_nf.job_creator import job_creator_dict
from causal_nf.job_creator.helper import *

import causal_nf.utils.io as causal_io

parser = argparse.ArgumentParser()
parser.add_argument(
    "-grid_file", "--grid_file", type=str, default=None, help="Grid file"
)
parser.add_argument(
    "-cluster_file",
    "--cluster_file",
    type=str,
    default=os.path.join("grids", "cluster_cpu.yaml"),
    help="Cluster file",
)
parser.add_argument(
    "-format",
    "--format",
    type=str,
    default="sub",
    help="Format to create the executable files",
)
parser.add_argument(
    "-wandb_mode",
    "--wandb_mode",
    type=str,
    default="offline",
    help="Format to create the executable files",
)
parser.add_argument(
    "-project", "--project", type=str, default="Test", help="Wandb project"
)
parser.add_argument(
    "-jobs_per_file",
    "--jobs_per_file",
    type=int,
    default=1,
    help="How many files to generate",
)
parser.add_argument(
    "-batch_size",
    "--batch_size",
    type=int,
    default=1,
    help="How many files to generate",
)
parser.add_argument("-delete_ckpt", "--delete_ckpt", action="store_true")
parser.add_argument("-only_test", "--only_test", action="store_true")

args = parser.parse_args()
grid = causal_io.load_yaml(args.grid_file, flatten=False)
grid_flat = causal_io.load_yaml(args.grid_file, flatten=True)
keys = list(grid_flat.keys())

folder = os.path.dirname(args.grid_file)
grid_name = os.path.basename(args.grid_file)
grid_file_extra_list = get_grid_file_extra_list(args.grid_file)

if len(grid_file_extra_list) > 0:
    options = []
    for grid_extra_i in grid_file_extra_list:
        causal_io.print_info(f"Getting configs from: {grid_extra_i}")
        options_i = generate_options(grid_flat=grid_flat, grid_file_extra=grid_extra_i)
        options.extend(options_i)
else:
    options = generate_options(grid_flat=grid_flat, grid_file_extra=None)

grid_folder = os.path.splitext(args.grid_file)[0]

causal_io.makedirs_rm_exist(grid_folder)
group = grid_folder.split(os.sep)[-2]
sub_folder = os.path.join(grid_folder, "jobs")
output_folder = os.path.join(grid_folder, "output")
config_folder = os.path.join(grid_folder, "configs")
scripts_folder = os.path.join(grid_folder, "scripts")

causal_io.makedirs(scripts_folder, only_if_not_exists=True)

job_creator = job_creator_dict[args.format](
    job_folder=sub_folder, output_folder=output_folder, header_file=args.cluster_file
)
num_jobs = len(options)
n_jobs_per_folder = args.jobs_per_file

main_str_list = []
folder_id_list = []
job_id_list = []

if args.only_test:
    options = options[:1]
causal_io.print_info(f"Number of jobs: {len(options)}")
for i, option in enumerate(options):
    folder_id = int(i // n_jobs_per_folder + 1)
    grid_folder_i = os.path.join(config_folder, str(folder_id))
    causal_io.makedirs(grid_folder_i, only_if_not_exists=True)
    cfg_i = causal_io.create_yaml(grid, keys, option)
    config_file = os.path.join(grid_folder_i, f"config_{i + 1}.yaml")
    causal_io.save_yaml(cfg_i, config_file)
    main_str = f"main.py --config_file {config_file}"
    main_str += f" --wandb_mode {args.wandb_mode}"
    main_str += f" --wandb_group {group}"
    main_str += f" --project {args.project}"
    if args.delete_ckpt:
        main_str += f" --delete_ckpt"
    job_id = int(i % n_jobs_per_folder)
    main_str_list.append(main_str)
    folder_id_list.append(folder_id)
    job_id_list.append(job_id)

i = 0
batch_job_id = 0
batch_main_str = os.path.join(scripts_folder, f"batch_{batch_job_id}.py")
causal_io.str_to_file(f"import os", batch_main_str)

num_jobs = len(main_str_list)
for main_str, folder_id, job_id in zip(main_str_list, folder_id_list, job_id_list):
    i += 1
    if args.batch_size == 1:
        job_creator.add_job(main_str, folder_id, job_id)
    else:
        causal_io.str_to_file(f"os.system('python {main_str}')", batch_main_str)

        if i % args.batch_size == 0 or i == num_jobs:
            job_creator.add_job(batch_main_str, folder_id, batch_job_id)
            batch_job_id += 1
            script_str = ""
            batch_main_str = os.path.join(scripts_folder, f"batch_{batch_job_id}.py")
            causal_io.str_to_file(f"import os", batch_main_str)

print(f"Total number of jobs: {i}")
