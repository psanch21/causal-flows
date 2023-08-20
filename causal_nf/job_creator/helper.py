import itertools
import os.path

import causal_nf.utils.io as causal_io


def get_value(value):
    if isinstance(value, str):
        value = eval(value)
    return value


def generate_options(grid_flat, grid_file_extra=None):
    values = []
    grid_flat_extra = None
    if isinstance(grid_file_extra, str) and os.path.exists(grid_file_extra):
        grid_flat_extra = causal_io.load_yaml(grid_file_extra, flatten=True)

    for key, value in grid_flat.items():
        if isinstance(value, str):
            assert value == "TODO", f"value: {value}"
            value = grid_flat_extra[key]

        value = get_value(value)
        assert isinstance(value, list), f"key | value: {key} | {value}"
        values.append(value)
    options = list(itertools.product(*values))
    return options


def get_grid_file_extra_list(grid_file):
    folder = os.path.dirname(grid_file)
    grid_basename = os.path.basename(grid_file)
    grid_name = os.path.splitext(grid_basename)[0]
    grid_list = []

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        cond1 = grid_basename != file
        cond2 = grid_name in file
        cond2 = grid_name == file[: len(grid_name)]
        cond3 = os.path.isfile(file_path)
        if cond1 and cond2 and cond3:
            grid_list.append(file_path)
    return grid_list
