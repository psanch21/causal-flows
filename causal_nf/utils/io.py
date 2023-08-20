import ast
import json
import os
import pickle
import shutil

import numpy as np
import torch
import yaml
from yacs.config import CfgNode as CN
import re


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class PRINT_CONFIG:
    WARNING = True
    DEBUG = True


def print_info(my_str):
    print(f"{bcolors.OKBLUE}[INFO] {my_str}{bcolors.ENDC}")


def print_warning(my_str):
    if PRINT_CONFIG.WARNING:
        print(f"{bcolors.WARNING}[WARNING] {my_str}{bcolors.ENDC}")


def print_debug(my_str):
    if PRINT_CONFIG.DEBUG:
        print(f"{bcolors.OKCYAN}[DEBUG] {my_str}{bcolors.ENDC}")


def print_empty_lines(n):
    print(
        "".join(
            [
                "\n",
            ]
            * n
        )
    )


def print_debug_tensor(tensor, name):
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor)
        print_debug(f"{name} is {type(tensor)}")
    if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
        mean_ = tensor.float().mean()
        print_debug(
            f"{name} [{tensor.dtype}] ({tensor.device}) | {tensor.shape} {tensor.min()}/{mean_}/{tensor.max()}"
        )
    else:
        print_debug(f"{name} is empty")


def print_dict(my_dict, title=None):
    if title is not None:
        print(f"{title}:")
    # Print a dictionary sorted by key
    for key, value in sorted(my_dict.items()):
        print(f"\t{key}: {value}")


def print_list(my_list, title=None):
    if title is not None:
        print(f"{title}:")
    for i, item in enumerate(my_list):
        print(f"\t{i}: {item}")


def string_to_python(string):
    try:
        return ast.literal_eval(string)
    except:
        return string


def dict_to_cn(my_dict):
    my_cn = CN()

    for key, value in my_dict.items():
        my_cn[key] = value

    return my_cn


def dict_to_json(dict, fname):
    with open(fname, "a") as f:
        json.dump(dict, f)
        f.write("\n")


def str_to_file(my_str, fname):
    with open(fname, "a") as f:
        f.write(f"{my_str}\n")


def dict_list_to_json(dict_list, fname):
    with open(fname, "a") as f:
        for dict in dict_list:
            json.dump(dict, f)
            f.write("\n")


def json_to_dict_list(fname, check_for_epoch=True, split=None):
    dict_list = []
    epoch_set = set()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            dict = json.loads(line)
            if isinstance(split, str):
                keys = list(dict.keys())
                append_line = False
                for key in list(dict.keys()):
                    if split in key:
                        append_line = True
                        break

                if not append_line:
                    continue

            if not check_for_epoch:

                dict_list.append(dict)
            else:
                if dict["epoch"] not in epoch_set:
                    dict_list.append(dict)
            epoch_set.add(dict["epoch"])
    return dict_list


def json_to_dict_list_simple(fname):
    dict_list = []
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            dict = json.loads(line)

            dict_list.append(dict)
    return dict_list


def dict_to_tb(dict, writer, epoch):
    for key in dict:
        writer.add_scalar(key, dict[key], epoch)


def dict_list_to_tb(dict_list, writer):
    for dict in dict_list:
        assert "epoch" in dict, "Key epoch must exist in stats dict"
        dict_to_tb(dict, writer, dict["epoch"])


def makedirs(dir, only_if_not_exists=False):
    if only_if_not_exists:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
    else:
        os.makedirs(dir, exist_ok=True)


def makedirs_rm_exist(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


def load_json(json_file):
    with open(json_file) as file:
        info = json.load(file)
    return info


def save_pickle(obj, pickle_file):
    with open(pickle_file, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(pickle_file):
    with open(pickle_file, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_yaml(my_dict, yaml_file):
    with open(yaml_file, "w") as f:
        documents = yaml.dump(my_dict, f)


def convert_to_none(my_config, inplace=True):
    assert inplace
    for key1, value1 in my_config.items():

        if isinstance(value1, dict):
            for key2, value2 in value1.items():
                if isinstance(value2, str) and value2.lower() == "none":
                    my_config[key1][key2] = None
        else:
            if isinstance(value1, str) and value1.lower() == "none":
                my_config[key1] = None


def load_yaml(yaml_file, flatten=False, unflatten=False, as_cn=False):
    with open(yaml_file) as file:
        my_yaml = yaml.load(file, Loader=yaml.FullLoader)

    convert_to_none(my_yaml)

    if flatten:
        my_yaml_ = {}
        for key1, value1 in my_yaml.items():

            assert "__" not in key1
            if isinstance(value1, dict):
                for key2, value2 in value1.items():
                    assert "__" not in key2
                    my_yaml_[f"{key1}__{key2}"] = value2
            else:
                my_yaml_[f"{key1}"] = value1

        my_yaml = my_yaml_

    if unflatten:
        my_yaml_ = {}
        for key, value in my_yaml.items():
            assert not isinstance(value, dict), f"{key} {value}"
            keys = key.split("__")
            if len(keys) == 1:
                my_yaml_[key] = value
            else:
                key1, key2 = keys
                if key1 not in my_yaml_:
                    my_yaml_[key1] = {}
                my_yaml_[key1][key2] = value
        my_yaml = my_yaml_

    if as_cn:
        my_cn = CN()
        for key, values in my_yaml.items():
            if isinstance(values, dict):
                my_cn[key] = dict_to_cn(values)
            else:
                my_cn[key] = values

        return my_cn
    else:
        return my_yaml


def create_yaml(base_grid, keys, option):
    cfg_i = base_grid.copy()
    for j, key in enumerate(keys):
        if "__" not in key:
            cfg_i[key] = option[j]
        else:
            key1, key2 = key.split("__")
            cfg_i[key1][key2] = option[j]
    return cfg_i


def save_delete(filename):
    if os.path.exists(filename):
        # Ask for confirmation before deleting
        print_warning(f"Delete {filename}?")
        delete = input("Y/n: ")
        if delete == "Y":
            print_warning(f"Deleting {filename}")
            os.remove(filename)
        else:
            print_info(f"Keeping {filename}")


def list_folders(path, add_default=None, regex_filter=None):
    folders = []
    for f in list(os.listdir(path)):
        if os.path.isdir(os.path.join(path, f)):
            if regex_filter is not None:
                if re.search(regex_filter, f):
                    folders.append(f)
            else:
                folders.append(f)

    if isinstance(add_default, str):
        folders.insert(0, add_default)

    return folders
