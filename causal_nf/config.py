import argparse
import collections
import os

from yacs.config import CfgNode as CN

from causal_nf.utils.io import load_yaml, print_warning

import causal_nf.utils.wandb_local as wandb_local

cfg = CN()

DEFAULT_CONFIG_FILE = os.path.join("causal_nf", "configs", "default_config.yaml")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config_file",
        "--config_file",
        type=str,
        default=None,
        help="Configuration_file",
    )
    parser.add_argument(
        "-config_default_file",
        "--config_default_file",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help="Configuration_file",
    )
    parser.add_argument(
        "-project", "--project", type=str, default=None, help="Project name"
    )
    parser.add_argument(
        "-wandb_mode",
        "--wandb_mode",
        type=str,
        default="disabled",
        help="mode of wandb",
        choices=["online", "offline", "run", "disabled", "dryrun"],
    )
    parser.add_argument(
        "-wandb_group", "--wandb_group", type=str, default=None, help="Group of wandb"
    )
    parser.add_argument(
        "-load_model", "--load_model", type=str, default=None, help="Load model"
    )
    parser.add_argument("-delete_ckpt", "--delete_ckpt", action="store_true")

    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    my_args = CN()
    args_list = []
    if isinstance(args.load_model, str):
        c_file, c_d_file = wandb_local.get_configs_from_folder(args.load_model)

        args.config_file = c_file
        if os.path.exists(c_d_file):
            args.config_default_file = c_d_file
        else:
            args.config_default_file = DEFAULT_CONFIG_FILE

    for key, value in args.__dict__.items():
        if key in [
            "config_file",
            "config_default_file",
            "wandb_mode",
            "wandb_group",
            "project",
            "load_model",
            "delete_ckpt",
        ]:
            my_args[key] = value
            continue
        if value is None:
            continue

        if key == "opts" and len(value) > 0:
            for i in range(0, len(value), 2):
                args_list.extend([value[i], value[i + 1]])

    return args_list, my_args


def get_cfg():
    return cfg


def update_cfg_from_dict(cfg, my_dict):
    for key, value in my_dict.items():
        if not hasattr(cfg, key):
            cfg[key] = CN()
        if isinstance(value, dict):
            for key2, value2 in value.items():
                if not hasattr(cfg[key], key2):
                    cfg[key][key2] = CN()
                cfg[key][key2] = value2
        else:
            cfg[key] = value


def get_config_default_file(as_dict=True, config_default_file=None):
    if config_default_file is None:
        my_config = load_yaml(DEFAULT_CONFIG_FILE)
    else:
        my_config = load_yaml(config_default_file)

    if not as_dict:
        my_cfg = CN()
        for key1, value1 in my_config.items():

            if isinstance(value1, dict):
                my_cfg[key1] = CN()
                for key2, value2 in value1.items():
                    my_cfg[key1][key2] = value2
            else:
                my_cfg[key1] = value1

        return my_cfg
    return my_config


def build_config(config_file=None, args_list=None, config_default_file=None):
    cfg_default = get_config_default_file(config_default_file=config_default_file)

    update_cfg_from_dict(cfg, cfg_default)

    if isinstance(config_file, str):
        if os.path.exists(config_file):
            assert_config_file(config_file)
            cfg.merge_from_file(config_file)
        else:
            print_warning(f"Config file does not exist: {config_file}")
            raise NotImplementedError

    if isinstance(args_list, list) and len(args_list) > 0:
        cfg.merge_from_list(args_list)

    config = collections.OrderedDict()
    for key1, value1 in cfg.items():

        assert "__" not in key1
        if isinstance(value1, dict):
            for key2, value2 in value1.items():
                assert "__" not in key2
                config[f"{key1}__{key2}"] = value2
        else:
            config[f"{key1}"] = value1

    return config


def assert_config_file(config_file):
    cfg_default = get_config_default_file()
    config = load_yaml(config_file)

    for key, value in config.items():
        assert key in cfg_default, f"key: {key}"
        if isinstance(value, dict):
            for key2, value2 in value.items():
                assert key2 in cfg_default[key], f"key: {key} | {key2}"


def assert_cfg_and_config(cfg, config):
    cfg_default = get_config_default_file()

    ## Assert types
    for key, value in cfg_default.items():
        value_cfg = None
        key_cfg = key
        if isinstance(value, dict):
            for key2, value2 in value.items():
                value = value2
                value_cfg = cfg[key][key2]
                key_cfg = f"{key}.{key2}"
        else:
            value_cfg = cfg[key]

        if type(value) != type(value_cfg):
            if value is None and not isinstance(value_cfg, str):
                raise ValueError(
                    f"Wrong datatype for {key_cfg}! cfg_default: {type(value)} cfg: {type(value_cfg)}"
                )

    ## Assert confgi and cfg has the same information
    for key, value in config.items():
        if "__" in key:
            key1, key2 = key.split("__")
            assert (
                config[key] == cfg[key1][key2]
            ), f"{key} {key1} {key2} |  {config[key]} {cfg[key1][key2]}"
        else:
            assert config[key] == cfg[key], f"{key}"

    assert cfg.train.batch_size > 0


def print_config(config):
    prev_key = None
    for key, value in config.items():
        if "__" not in key:
            print(f"{key}: {value} ({type(value)})")
        else:
            key1, key2 = key.split("__")
            if key1 != prev_key:
                print(f"{key1}:")
                prev_key = key1
            print(f"  {key2}: {value} ({type(value)})")
