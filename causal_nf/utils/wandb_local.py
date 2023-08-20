import glob
import os
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd

import causal_nf.utils.io as causal_io


def get_configs_from_folder(folder):
    wandb_local_folder = os.path.join(folder, 'wandb_local')
    config_file = os.path.join(wandb_local_folder, 'config_local.yaml')
    if True: # not os.path.exists(config_file):
        config_file_ = os.path.join(wandb_local_folder, 'config.yaml')
        config = causal_io.load_yaml(yaml_file=config_file_, unflatten=True)
        causal_io.save_yaml(config, yaml_file=config_file)

    config_default_file = os.path.join(wandb_local_folder, 'default_config.yaml')
    return config_file, config_default_file


def logs_folder(root):
    folder = os.path.join(root, 'wandb_local')
    return folder


def sub_folder(root, subfolder):
    folder = logs_folder(root)
    my_subfolder = os.path.join(folder, subfolder)
    causal_io.makedirs(my_subfolder, only_if_not_exists=True)
    return my_subfolder


def list_wandb_local_folders(root_dir):
    folders = []
    for folder in glob.glob(os.path.join(root_dir, '**/'), recursive=True):
        if folder[-1] == os.sep: folder = folder[:-1]
        if 'wandb_local' == folder.split(os.sep)[-1]:
            folders.append(folder)
    return folders


def load_experiments_from_wandb_folders(wandb_folders):
    df_list = []
    for folder in wandb_folders:
        try:
            df = load_experiment_v2(folder=folder)
            df.reset_index(drop=True, inplace=True)
            df_list.append(df)
        except Exception:
            causal_io.print_warning(f"Failed to load {folder}!")

    df = pd.concat(df_list)

    df.reset_index(inplace=True)

    return df


def replace_nan_by_default_value(df, inplace=True):
    assert inplace
    default_values = {
        np.float64: 0.0,
        np.int64: 0,
        np.bool_: False,
        np.object_: ''
    }
    for col in df.columns:
        col_type = df[col].dtype.type
        if col_type in default_values:
            default_value = default_values[col_type]
            df[col] = df[col].fillna(default_value)


def load_experiments(root_dir,
                     max_number=0,
                     remove_nan=False,
                     add_timestamp=False):
    wandb_folders = list_wandb_local_folders(root_dir)
    if max_number > 0:
        wandb_folders = wandb_folders[:max_number]
    df = load_experiments_from_wandb_folders(wandb_folders)
    if remove_nan:
        df = df[~df.dataset__root.isna()]
    if add_timestamp:
        df['datetime'] = df.timestamp.apply(lambda x: datetime.fromtimestamp(x))
    return df


def str_to_dict(my_str, sep='--', remove_ext=False):
    if remove_ext:
        my_str = os.path.splitext(my_str)[0]
    my_dict = {}
    elments = os.path.splitext(my_str)[0].split(sep)
    for el in elments:
        if '=' in el:
            key, value = el.split('=')
            my_dict[key] = value
        else:
            my_dict['name'] = el
    return my_dict


def dict_to_str(my_dict, keys=None, keys_remove=None):
    if keys is None:
        keys = list(my_dict.keys())

    if keys_remove is not None:
        keys = list(set(keys) - set(keys_remove))

    keys = sorted(keys)
    my_str = []
    for key in keys:
        if key in my_dict:
            my_str.append(f"{key}={my_dict[key]}")

    return '__'.join(my_str)


def load_images_experiment(folder, folder_name='images'):
    images_folder = os.path.join(folder, folder_name)
    images = os.listdir(images_folder)
    images_list = []

    for im in images:
        im_dict = {'filename': os.path.join(images_folder, im)}
        im_dict.update(str_to_dict(my_str=im))
        images_list.append(im_dict)

    df = pd.DataFrame.from_records(images_list)
    return df


def load_experiment(folder):
    file_logs = os.path.join(folder, 'logs.txt')

    splits = ['train', 'val', 'test']
    df_list = []
    for split in splits:

        logs_list = causal_io.json_to_dict_list(file_logs,
                                              check_for_epoch=False,
                                              split=split)
        if len(logs_list) == 0: continue
        df_split = pd.DataFrame.from_records(logs_list)

        df_split = df_split.drop_duplicates(subset='epoch', keep='last')
        df_split.set_index(keys='epoch', inplace=True)
        df_list.append(df_split)

    df = pd.concat(df_list, axis=1)
    add_config_columns(df=df, folder=folder, inplace=True)

    return df


def load_experiment_v2(folder):
    file_logs = os.path.join(folder, 'logs.txt')

    logs_list = causal_io.json_to_dict_list(file_logs, check_for_epoch=False)
    records = []

    for log_i in logs_list:
        record_i = {}
        for key, value in log_i.items():
            if isinstance(value, dict):
                record_i['split'] = key
                for key2, value2 in value.items():
                    record_i[key2] = value2
            else:
                record_i[key] = value

        records.append(record_i)

    df = pd.DataFrame.from_records(records)

    df = df.drop_duplicates(subset=['epoch', 'split'], keep='last')

    add_config_columns(df=df, folder=folder, inplace=True)

    return df


def add_config_columns(df, folder, inplace=True):
    file_config = os.path.join(folder, 'config.yaml')

    try:
        config_dict = causal_io.load_yaml(file_config)

        assert inplace
        for key, value in config_dict.items():
            if isinstance(value, list):
                df[key] = [value, ] * len(df)
            else:
                df[key] = value
    except Exception:
        causal_io.print_warning(f"Failed to load {file_config}!")

    folder_split = folder.split(os.sep)
    if folder_split[-1] == '':
        df['experiment_name'] = folder_split[-3]
    else:
        df['experiment_name'] = folder_split[-2]


def log_config(config, root):
    folder = logs_folder(root)
    causal_io.makedirs(folder, only_if_not_exists=True)

    filename = os.path.join(folder, 'config.yaml')
    causal_io.save_yaml(config, yaml_file=filename)


def copy_config(config_experiment, root, config_default):

    folder = logs_folder(root)
    causal_io.makedirs(folder, only_if_not_exists=True)

    basename_default = os.path.basename(config_default)
    filename = os.path.join(folder, basename_default)
    shutil.copyfile(config_default, filename)

    filename = os.path.join(folder, 'config_local.yaml')
    shutil.copyfile(config_experiment, filename)


def log(metrics_dict, root):
    folder = logs_folder(root)
    causal_io.makedirs(folder, only_if_not_exists=True)

    filename = os.path.join(folder, 'logs.txt')

    output = {}
    timestamp_added = False

    for key, value in metrics_dict.items():
        if not isinstance(value, dict):
            output[key] = value
        else:
            for key2, value2 in value.items():
                output[f"{key}/{key2}"] = value2
                if not timestamp_added:
                    output[f'{key}/timestamp'] = int(time.time())
                    timestamp_added = True
    if not timestamp_added:
        output[f'timestamp'] = int(time.time())

    causal_io.dict_to_json(output, fname=filename)


def log_v2(metrics_dict, root):
    metrics_dict_print = {}
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                if isinstance(value2, float):
                    metrics_dict_print[key2] = f"{value2:.3f}"
                else:
                    metrics_dict_print[key2] = value2
        else:
            if isinstance(value, float):
                metrics_dict_print[key] = f"{value:.3f}"
            else:
                metrics_dict_print[key] = value

    if isinstance(root, str):
        folder = logs_folder(root)
        causal_io.makedirs(folder, only_if_not_exists=True)

        filename = os.path.join(folder, 'logs.txt')
        metrics_dict['timestamp'] = int(time.time())
        causal_io.dict_to_json(metrics_dict, fname=filename)
    else:
        causal_io.print_warning(f"Not logging, root is None.")


def print_parameters_cv(df, remove_nan=True):
    causal_io.print_warning(f"See parameters_cross_validated in dataframe.py")
    raise NotImplementedError


def get_number_experiments_per_folder(root):
    i = 0
    folder_count = {}
    folders = causal_io.list_folders(root)
    for folder in folders:
        folder_path = os.path.join(root, folder)
        experiments = []

        for folder_exp in os.listdir(folder_path):
            exp_path = os.path.join(folder_path, folder_exp)
            if os.path.isdir(exp_path):
                experiments.append(exp_path)

        folder_count[folder_path] = len(experiments)

    return folder_count
