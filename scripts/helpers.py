import seaborn as sns
import hashlib
import os
from causal_nf.utils.struct import Struct
import pandas as pd

import causal_nf.utils.wandb_local as wandb_local

from datetime import datetime

palette_name = "colorblind"

ext = "pdf"

sns.set_palette(palette_name)
# Get the colors from the palette
colors = sns.color_palette(palette_name)

import re


def remove_non_alphanumeric(string):
    return re.sub(r"\W+", "", string).lower()


def select_color(model):
    adj = "Causal" in model
    if not adj:
        return colors[2]
    else:
        return colors[3]


def select_marker(model):
    if "*" in model:
        return "*"
    else:
        return "."


def select_style(model):
    if "u-x" in model:
        return ":"
    else:
        return "-"


ticks_formatter = {}


def format_ticks(x, pos):
    return f"{x:.2f}"  # Format to two decimal places


ticks_formatter["loss_jacobian_x"] = format_ticks


def format_ticks(x, pos):
    return x  # Format to two decimal places


ticks_formatter["kl_forward"] = format_ticks


def format_ticks(x, pos):
    return f"{x:.2f}"  # Format to two decimal places


ticks_formatter["rmse_ate"] = format_ticks


def _update_names(df, new_name, level_name_list, mapping=None):
    df_tmp = df.copy()

    new_column = None
    for i, level_name_i in enumerate(level_name_list):
        if i == 0:
            new_column = df_tmp[level_name_i].astype(str).values
        else:
            new_column += "-" + df_tmp[level_name_i].astype(str).values

    df_tmp[new_name] = new_column
    if isinstance(mapping, dict):
        df_tmp[new_name] = df_tmp[new_name].replace(mapping)

    return df_tmp


def update_model_name_vaca(df):
    df["model__name"].replace({"vaca": "VACA"}, inplace=True)
    return df


def update_model_name_flow(df):
    mapping = {}
    for flow_name in ["maf"]:
        for use_adj in ["True", "False"]:
            for reg in ["True", "False"]:
                pre_str = "Causal" if use_adj == "True" else ""
                values = pre_str + flow_name.upper()
                if reg == "True":
                    values += "*"
                mapping[f"{flow_name}-{use_adj}-{reg}"] = values

    df_tmp = _update_names(
        df=df,
        new_name="model__name",
        level_name_list=["model__layer_name", "model__adjacency", "train__regularize"],
        mapping=mapping,
    )

    return df_tmp


def update_names(df, column_mapping=None, model="flow"):
    df_tmp = df.copy()
    if model == "flow":
        df_tmp = update_model_name_flow(df_tmp)
    elif model == "vaca":
        df_tmp = update_model_name_vaca(df_tmp)

    df_tmp["dataset__name"].replace(
        {"chain": "chain-3", "large-backdoor": "large-bd"}, inplace=True
    )

    mapping = {}
    mapping["linear"] = "LIN"
    mapping["non-linear"] = "NLIN"
    mapping["sym-prod"] = "NLIN2"
    df_tmp["dataset__sem_name"] = df_tmp["dataset__sem_name"].map(mapping)
    mapping = {}

    for dataset_name in df_tmp.dataset__name.unique():
        for sem_name in df_tmp.dataset__sem_name.unique():
            part_1 = f"{dataset_name.upper()}"
            part_2 = f"{sem_name.upper()}"
            mapping[f"{dataset_name}-{sem_name}"] = f"{part_1}[{part_2}]"

    df_tmp = _update_names(
        df=df_tmp,
        new_name="dataset__name",
        level_name_list=["dataset__name", "dataset__sem_name"],
        mapping=mapping,
    )

    mapping = {}

    mapping[True] = "u-x"
    mapping[False] = "x-u"

    df_tmp["model__base_to_data"] = df_tmp["model__base_to_data"].map(mapping)

    if column_mapping == None:
        column_mapping = {}

        column_mapping["dataset__name"] = "Dataset"
        column_mapping["model__name"] = "Model"
        column_mapping["model__base_to_data"] = "Direction"

        column_mapping["model__num_layers"] = "$L$"

    df_tmp["model__dim_inner"] = df_tmp["model__dim_inner"].apply(
        lambda x: "-".join([str(s) for s in eval(x)])
    )

    df_tmp = df_tmp.rename(columns=column_mapping)

    return df_tmp


def load_df(root, exp_folders, keep_cols=None, freq=1):
    exp_folders_str = " ".join(exp_folders)
    exp_folders_hash = hashlib.md5(exp_folders_str.encode()).hexdigest()
    df_filename = os.path.join(root, f"df_{exp_folders_hash}.pkl")

    print(f"df_filename: {df_filename}")

    # Create dataframe

    df = Struct()

    if os.path.exists(df_filename):
        timestamp = os.path.getmtime(df_filename)
        # convert the timestamp to a datetime object
        last_modified = datetime.fromtimestamp(timestamp)
        print(f"Loading {df_filename} [{last_modified}]")
        df_all = pd.read_pickle(df_filename)
    else:

        df_list = []
        if any([exp_folder == "ALL" for exp_folder in exp_folders]):
            exp_folders = ["ALL"]
        for exp_folder in exp_folders:
            if exp_folder != "ALL":
                root_dir = os.path.join(root, exp_folder)
            else:
                root_dir = root

            df_all_i = wandb_local.load_experiments(
                root_dir=root_dir, max_number=0, remove_nan=True, add_timestamp=True
            )
            df_list.append(df_all_i)

        df_all = pd.concat(df_list)

        if keep_cols is None:
            keep_cols = []
            keep_cols.append("dataset__name")
            keep_cols.append("model__name")
            keep_cols.append("model__layer_name")

        df_training = df_all[~df_all.epoch.isin(["best", "last"])]
        df_training = df_training[df_training.epoch % freq == 0]

        df_best = df_all[df_all.epoch == "best"]
        df_last = df_all[df_all.epoch == "last"]

        df_all = pd.concat([df_best, df_last, df_training])

        for column in df_all.columns:
            delete = False
            try:
                if len(df_all[column].unique()) == 1:
                    delete = True
            except:
                print(f"\tERROR: {column}")
                df_all[column] = df_all[column].apply(str)
                if len(df_all[column].unique()) == 1:
                    delete = True

            if delete and column not in keep_cols:
                df_all = df_all.drop(column, axis=1)

        df_all.to_pickle(df_filename)

    memory_usage_mb = df_all.memory_usage().sum() / (1024 * 1024)
    print(f"Memory usage of dataframe: {memory_usage_mb:.2f} MB")
    print(f"df_all: {df_all.shape}")

    df.all = df_all
    df.training = df_all[~df_all.epoch.isin(["best", "last"])]
    df.best = df_all[df_all.epoch == "best"]
    df.last = df_all[df_all.epoch == "last"]
    return df
