#
import sys

sys.path.append("../")
sys.path.append("./")

import os

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

import notebooks.helpers as pb_help

import matplotlib.pyplot as plt

from tueplots import bundles

plt.rcParams.update(bundles.icml2022())
# plt.rcParams.update(figsizes.icml2022_full())
import scripts.helpers as script_help

import causal_nf.utils.dataframe as causal_nf_df
import warnings

import causal_nf.utils.io as causal_nf_io

warnings.filterwarnings("ignore")

root = "output_causal_nf"
folder = os.path.join("results", "dataframes")
causal_nf_io.makedirs(folder, only_if_not_exists=True)
keep_cols = []
keep_cols.append("dataset__name")
keep_cols.append("dataset__sem_name")
keep_cols.append("dataset__num_samples")
keep_cols.append("dataset__base_version")

keep_cols.append("model__name")
keep_cols.append("model__layer_name")
keep_cols.append("model__dim_inner")
keep_cols.append("model__num_layers")
keep_cols.append("model__adjacency")
keep_cols.append("model__base_to_data")
keep_cols.append("model__base_distr")

keep_cols.append("train__regularize")

# %% Load dataframes
exp_folders = [
    "comparison_x_u",
    "comparison_u_x",
    "comparison_carefl",
    # 'comparison_largebd'
]
df_all_last_list = []
df_all_training_list = []
for exp_folder in exp_folders:
    df = pb_help.load_df(root, [exp_folder], keep_cols, freq=10)
    df_all_last_list.append(df.last)
    df_all_training_list.append(df.training)

df_last_all = pd.concat(df_all_last_list, axis=0)
df_training_all = pd.concat(df_all_training_list, axis=0)

# %%

column_mapping = {}
column_mapping["dataset__name"] = "Dataset"
column_mapping["dataset__sem_name"] = "SEM"
column_mapping["model__name"] = "Model"
column_mapping["model__base_to_data"] = "Direction"

# %% Get training

filter_ = {}
filter_["model__base_to_data"] = [False]

df_ = causal_nf_df.filter_df(df_training_all.copy(), filter_)
df_ = df_[df_.split == "train"]
df_ = df_[~df_.epoch.isin(["last", "best"])]
df_ = script_help.update_names(df_, column_mapping=column_mapping)

row_id_cols = ["SEM", "Dataset", "Model", "model__dim_inner"]

row_id_cols_all = [*row_id_cols, "time_forward"]

df_training = df_[row_id_cols_all].groupby(row_id_cols).agg(["mean", "std", "count"])

# %% Get last

filter_ = {}
filter_["model__base_to_data"] = [False]

df_ = causal_nf_df.filter_df(df_last_all.copy(), filter_)
df_ = script_help.update_names(df_, column_mapping=column_mapping)

df_["rmse_cf"] = df_.filter(regex="rmse_cf").mean(1)
df_["mmd_int"] = df_.filter(regex="mmd_int").mean(1)
df_["rmse_ate"] = df_.filter(regex="rmse_ate").mean(1)
df_["kl_forward"] = df_["log_prob_true"] - df_["log_prob"]

row_id_cols = ["SEM", "Dataset", "Model"]

row_id_cols_ = [
    *row_id_cols,
]

df_best_all, row_id_cols_all = pb_help.get_best_df(
    df=df_, row_id_cols=row_id_cols, fn=np.argmax, metric="log_prob", show=False
)

df_tmp = df_best_all.groupby(row_id_cols_all).agg(["mean", "std", "count"])

# %%


df_table = df_tmp.copy()
cond = df_table.index.get_level_values("split") == "test"
df_table = df_table.loc[cond, :]

# Remove the specified level names
for level in ["model__num_layers", "split"]:
    df_table.index = df_table.index.droplevel(level)

# df_table = df_table.droplevel(level=[0,2])
l0 = df_table.index.names.index("SEM")
l1 = df_table.index.names.index("Dataset")
l2 = df_table.index.names.index("Model")
l3 = df_table.index.names.index("model__dim_inner")

df_table.index = df_table.index.reorder_levels([l0, l1, l2, l3])

l0 = df_training.index.names.index("SEM")
l1 = df_training.index.names.index("Dataset")
l2 = df_training.index.names.index("Model")
l3 = df_training.index.names.index("model__dim_inner")
df_training.index = df_training.index.reorder_levels([l0, l1, l2, l3])

df_training_good = df_training.loc[df_table.index]
df_table = df_table.drop(["time_forward"], axis=1)
df_table = df_table.join(df_training_good)

# Get the list of level names to remove
level_names_to_remove = [
    level for level in df_table.index.names if level not in ["Model", "Dataset", "SEM"]
]

# Remove the specified level names
for level in level_names_to_remove:
    df_table.index = df_table.index.droplevel(level)

l0 = df_table.index.names.index("SEM")
l1 = df_table.index.names.index("Dataset")
l2 = df_table.index.names.index("Model")

df_table = df_table.sort_index(level=[l0, l1, l2], ascending=True)

metrics_cols = [
    "time_sample_obs",
    "time_log_prob",
    "time_forward",
    "kl_forward",
    "rmse_ate",
    "rmse_cf",
    "param_count",
]

df_table = df_table.loc[:, df_table.columns.get_level_values(0).isin(metrics_cols)]

# %%

filename = os.path.join(folder, "comparison_flows.pickle")
print("Saving to {}".format(filename))
df_table.to_pickle(filename)
