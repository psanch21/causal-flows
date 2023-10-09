import hashlib
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tueplots import bundles

import causal_nf.utils.dataframe as pb_df
import causal_nf.utils.io as pb_io
import causal_nf.utils.list_op as toro_list
import causal_nf.utils.string as pb_str
import causal_nf.utils.wandb_local as wandb_local
from causal_nf.utils.struct import Struct

plt.rcParams.update(bundles.icml2022())


def select_root(idx=0, regex_filter='output'):
    folders = pb_io.list_folders('.', add_default='', regex_filter=regex_filter)
    for i, f in enumerate(folders):
        print(f"[{i}] {f}")

    root = folders[idx]
    print(f"\nroot: {root}")
    return root


def select_subfolders(root, exp_folders_ids=None, exp_folders_names=None):
    # Count the number of experiments per folder
    folder_count = wandb_local.get_number_experiments_per_folder(os.path.join('.', root))
    for key, value in folder_count.items():
        print(f"{key}: {value}")

    subfolders = pb_io.list_folders(os.path.join('.', root), add_default='ALL')

    for i, subf in enumerate(subfolders):
        print(f"[{i}] {subf}")

    if exp_folders_ids is None:
        exp_folders_ids = list(range(len(subfolders)))

    if exp_folders_names is not None:
        exp_folders = sorted(exp_folders_names)
    else:
        exp_folders = sorted([subfolders[i] for i in exp_folders_ids])
    print(f"\nexp_folders: {exp_folders}")
    return exp_folders


def load_df(root, exp_folders, keep_cols=None, freq=1):
    exp_folders_str = ' '.join(exp_folders)
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
        if any([exp_folder == 'ALL' for exp_folder in exp_folders]):
            exp_folders = ['ALL']
        for exp_folder in exp_folders:
            if exp_folder != 'ALL':
                root_dir = os.path.join(root, exp_folder)
            else:
                root_dir = root

            df_all_i = wandb_local.load_experiments(root_dir=root_dir,
                                                    max_number=0,
                                                    remove_nan=True,
                                                    add_timestamp=True)
            df_list.append(df_all_i)

        df_all = pd.concat(df_list)

        if keep_cols is None:
            keep_cols = []
            keep_cols.append('dataset__name')
            keep_cols.append('model__name')
            keep_cols.append('model__layer_name')

        df_training = df_all[~df_all.epoch.isin(['best', 'last'])]
        df_training = df_training[df_training.epoch % freq == 0]

        df_best = df_all[df_all.epoch == 'best']
        df_last = df_all[df_all.epoch == 'last']

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

    # df_all = df_all[df_all.model__name == 'reindeer']
    # df_all = df_all[df_all.reward__name == 'ratio_ub_entropy_diff']

    memory_usage_mb = df_all.memory_usage().sum() / (1024 * 1024)
    print(f"Memory usage of dataframe: {memory_usage_mb:.2f} MB")
    print(f"df_all: {df_all.shape}")

    df.all = df_all
    df.training = df_all[~df_all.epoch.isin(['best', 'last'])]
    df.best = df_all[df_all.epoch == 'best']
    df.last = df_all[df_all.epoch == 'last']
    return df


def get_hyperparameters_cv(df, show=False):
    p_warning = pb_io.PRINT_CONFIG.WARNING
    pb_io.PRINT_CONFIG.WARNING = False
    params_dict = pb_df.parameters_cross_validated(df=df)
    if show: pb_io.print_dict(my_dict=params_dict, title=f"Hyperparameters CV")
    pb_io.PRINT_CONFIG.WARNING = p_warning
    return params_dict


def get_best_df(df, row_id_cols=None,
                fn=None,
                metric=None,
                show=True):
    df_best = []
    row_id_cols_all = [*row_id_cols, 'split']
    for row_id, df_row in df.groupby(row_id_cols):
        best_config = {}
        df_row_val = df_row[df_row.split == 'val'].copy()
        hyperparameters_cv = get_hyperparameters_cv(df_row_val, show=show)
        cols = list(hyperparameters_cv.keys())
        row_id_cols_all = toro_list.list_union(row_id_cols_all, cols)
        cols_all = [*cols, metric]
        df_g = df_row_val[cols_all].groupby(cols).agg(['mean', 'count']).reset_index()
        values = df_g[(metric, 'mean')]
        best_id = fn(values)
        row_best = df_g.iloc[best_id]
        print(f"{row_id}")
        df_out = df_row.copy()

        for c in cols:
            c_best = row_best[c].item()
            best_config[c] = c_best
            print(f"\t{c}[{c_best}]: {hyperparameters_cv[c]}")
            df_out = df_out[df_out[c] == c_best]

        df_best.append(df_out)

    df_best = pd.concat(df_best)
    return df_best, row_id_cols_all


def lineplots(df, y, x, col, row,
              hue=None,
              style=None, filename=None, y_lim=None,
              df_horizontal=None,
              y_log=False, show=True):
    num_cols = len(df[col].unique())

    num_rows = len(df[row].unique())

    figsize = list(bundles.icml2022()['figure.figsize'])
    figsize[0] *= num_cols
    figsize[1] *= num_rows
    figsize = tuple(figsize)

    hue_order = None if hue is None else sorted(df[hue].unique())

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    legend_row = -(-num_rows // 2) - 1

    if style:
        style_order = sorted(df[style].unique())
    else:
        style_order = None

    for i, (row_name, df_i) in enumerate(df.groupby(row)):
        for j, (col_name, df_ij) in enumerate(df_i.groupby(col)):
            ax = get_ax(axes, num_cols, num_rows, i, j)
            if isinstance(y, list):
                y_0, y_1 = y
            else:
                y_0 = y
            _ = sns.lineplot(data=df_ij,
                             x=x,
                             y=y_0,
                             hue=hue,
                             hue_order=hue_order,
                             style=style,
                             style_order=style_order,
                             ax=ax)

            if df_horizontal is not None:
                ax.axhline(df_horizontal.loc[row_name, y_0], color='black', linestyle='--')

            if isinstance(y, list):
                ax2 = ax.twinx()
                _ = sns.lineplot(data=df_ij,
                                 x=x,
                                 y=y_1,
                                 hue=hue,
                                 hue_order=hue_order,
                                 style=style,
                                 style_order=style_order,
                                 color='red',
                                 ax=ax2)

            ax.set_title(f"{row_name} - {col_name}")
            if y_log:
                ax.set_yscale("log")
                if isinstance(y, list):
                    ax2.set_yscale('log')

            if j == 0:
                if y_lim is None:
                    y_lim_ = ax.get_ylim()
                    diff = (y_lim_[1] - y_lim_[0]) * 0.15
                    y_lim_ = (y_lim_[0] - diff, y_lim_[1] + diff)
                else:
                    y_lim_ = y_lim

            ax.set_ylim(y_lim_)
            if isinstance(y, list):
                ax2.set_ylim(y_lim_)

            if i == legend_row and j == (num_cols - 1) and hue is not None:
                l = ax.legend(loc='center right', bbox_to_anchor=(1.4, 0.5), ncol=1)
                l.set_title(pb_str.capitalize_words(hue))
            elif hue is not None:
                ax.legend().remove()

            ax.grid(True)
            if isinstance(y, list):
                ax2.grid(True)

        # g.set(yscale='log')
        fig.suptitle(f"Rows: {row} - Cols: {col}")

    # Add a legend to the last axis
    if filename is not None:
        fig.savefig(filename)
    if show: plt.show()


def get_ax(axes, num_cols, num_rows, i, j):
    if num_cols == 1 and num_rows == 1:
        ax = axes[i]
    if num_cols == 1:
        ax = axes[i]
    elif num_rows == 1:
        ax = axes[j]
    else:
        ax = axes[i, j]
    return ax


def boxplots(df, y, x, col, row,
             hue=None,
             filename=None,
             show=True,
             y_lim=None,
             df_horizontal=None,
             return_df=False):
    num_cols = len(df[col].unique())

    num_rows = len(df[row].unique())

    figsize = list(bundles.icml2022()['figure.figsize'])
    figsize[0] *= num_cols
    figsize[1] *= num_rows
    figsize = tuple(figsize)

    order = sorted(df[x].unique())
    if hue is not None:
        hue_order = sorted(df[hue].unique())
    else:
        hue_order = None

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    legend_row = -(-num_rows // 2) - 1

    for i, (row_name, df_i) in enumerate(df.groupby(row)):
        for j, (col_name, df_ij) in enumerate(df_i.groupby(col)):

            ax = get_ax(axes, num_cols, num_rows, i, j)
            _ = sns.boxplot(data=df_ij,
                            x=x,
                            y=y,
                            hue=hue,
                            order=order,
                            hue_order=hue_order,
                            showfliers=False,
                            ax=ax)

            if df_horizontal is not None:
                ax.axhline(df_horizontal.loc[row_name, y], color='black', linestyle='--')

            ax.set_title(f"{row_name} - {col_name}")

            if j == 0:
                if y_lim is not None:
                    y_lim_i = y_lim
                else:
                    y_lim_i = ax.get_ylim()
                    diff = (y_lim_i[1] - y_lim_i[0]) * 0.15
                    y_lim_i = (y_lim_i[0] - diff, y_lim_i[1] + diff)

            ax.set_ylim(y_lim_i)

            if i == legend_row and j == (num_cols - 1):
                if hue is not None:
                    l = ax.legend(loc='center right', bbox_to_anchor=(1.4, 0.5), ncol=1)
                    l.set_title(pb_str.capitalize_words(hue))
            else:
                ax.legend().remove()

            ax.grid(True)

        # g.set(yscale='log')
        fig.suptitle(f"Rows: {row} - Cols: {col}")

    # Add a legend to the last axis
    if filename is not None:
        fig.savefig(filename)
    if show: plt.show()

    if return_df:
        if hue is None:
            groupby = [x, col, row]
        else:
            groupby = [x, col, row, hue]
        cols = [*groupby, y]
        df_out = df[cols].groupby(groupby).agg(['mean', 'std', 'count'])
        return df_out


def scatterplots(df, y, x, col, row, hue,
                 filename=None,
                 show=True,
                 y_lim=None,
                 return_df=False):
    num_cols = len(df[col].unique())

    num_rows = len(df[row].unique())

    figsize = list(bundles.icml2022()['figure.figsize'])
    figsize[0] *= num_cols
    figsize[1] *= num_rows
    figsize = tuple(figsize)

    order = sorted(df[x].unique())

    hue_order = sorted(df[hue].unique())

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    legend_row = -(-num_rows // 2) - 1

    for i, (row_name, df_i) in enumerate(df.groupby(row)):
        for j, (col_name, df_ij) in enumerate(df_i.groupby(col)):
            ax = axes[i, j]
            _ = sns.scatterplot(data=df_ij,
                                x=x,
                                y=y,
                                hue=hue,
                                order=order,
                                hue_order=hue_order,
                                showfliers=False,
                                ax=ax)

            ax.set_title(f"{row_name} - {col_name}")

            if j == 0:
                if y_lim is not None:
                    y_lim_i = y_lim
                else:
                    y_lim_i = ax.get_ylim()
                    diff = (y_lim_i[1] - y_lim_i[0]) * 0.15
                    y_lim_i = (y_lim_i[0] - diff, y_lim_i[1] + diff)

            ax.set_ylim(y_lim_i)

            if i == legend_row and j == (num_cols - 1):
                l = ax.legend(loc='center right', bbox_to_anchor=(1.4, 0.5), ncol=1)
                l.set_title(pb_str.capitalize_words(hue))
            else:
                ax.legend().remove()

            ax.grid(True)

        # g.set(yscale='log')
        fig.suptitle(f"Rows: {row} - Cols: {col}")

    # Add a legend to the last axis
    if filename is not None:
        fig.savefig(filename)
    if show: plt.show()

    if return_df:
        groupby = [x, col, row, hue]
        cols = [*groupby, y]
        df_out = df[cols].groupby(groupby).agg(['mean', 'std', 'count'])
        return df_out


def get_df(df, mode='last', show_cv=False, filter_=None):
    df_ = getattr(df, mode)
    if isinstance(filter_, dict):
        df_ = pb_df.filter_df(df_, filter_)

    if show_cv:
        cols = []
        for c in df_.columns:
            if '__' not in c: continue
            try:
                u = df_[c].unique()
                if len(u) > 1: print(f"{c}: {u}")
            except:
                pass

    return df_


def map_level(df, dct, level=0):
    index = df.index
    index.set_levels([[dct.get(item, item) for item in names] if i == level else names
                      for i, names in enumerate(index.levels)], inplace=True)


def combine_levels(df, new_name, level_name_list, new_level=0):
    df = df.copy()
    index = df.index

    values = None
    for level_name in level_name_list:
        values_i = index.get_level_values(level_name)
        if values is None:
            values = list(values_i)
        else:
            values = [f"{v}-{values_i[i]}" for i, v in enumerate(values)]

    df['new_col'] = values

    new_index = list(df.index.names)
    new_index.insert(new_level, 'new_col')
    df.reset_index(inplace=True)
    df.set_index(new_index, inplace=True)
    df = df.droplevel(level=level_name_list)
    df.index = df.index.set_names(new_name, level=new_level)
    return df
