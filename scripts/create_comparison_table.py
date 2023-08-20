#
import sys

sys.path.append("../")
sys.path.append("./")


import pandas as pd
import os

folder = os.path.join("results", "dataframes")

df_vaca = pd.read_pickle(os.path.join(folder, "comparison_vaca.pickle"))
df_flows = pd.read_pickle(os.path.join(folder, "comparison_flows.pickle"))

df_table = pd.concat([df_flows, df_vaca])


# %%

metrics_cols = [
    "time_sample_obs",
    "time_log_prob",
    "time_forward",
    "kl_forward",
    "rmse_ate",
    "rmse_cf",
]

data = []
# iterate over the rows and combine the mean and std values using string formatting
for row_id, row in df_table.iterrows():
    row_dict = {}
    row_dict["Dataset"] = row_id[1]
    row_dict["Model"] = row_id[2]
    cond_1 = df_table.index.get_level_values("SEM") == row_id[0]
    cond_2 = df_table.index.get_level_values("Dataset") == row_id[1]
    df_block = df_table.loc[cond_1 & cond_2]
    for metric in metrics_cols:

        mean = row[(metric, "mean")]
        std = row[(metric, "std")]
        mean_max = df_block[(metric, "mean")].min()
        is_best = mean_max > (mean - std)
        if is_best:
            # if std == 0.0:
            #     value = f"\\bfseries {mean:.2f}"
            # else:
            #     value = f"\\bfseries {mean:.2f}_{{{std:.2f}}}"
            if std == 0.0:
                value = f"\\mathbf{{{mean:.2f}}}"
            else:
                value = f"\\mathbf{{ {mean:.2f}_{{{std:.2f}}}}}"
        else:
            if std == 0.0:
                value = f"{mean:.2f}"
            else:
                value = f"{mean:.2f}_{{{std:.2f}}}"
        row_dict[metric] = value
    data.append(row_dict)

df_latex = pd.DataFrame(data)
df_latex = df_latex.set_index(["Dataset", "Model"])
df_latex = df_latex.sort_values(by=["Dataset", "Model"])


mapping = {}
mapping["kl_forward"] = "KL Forward"
mapping["rmse_ate"] = "RMSE ATE"
mapping["rmse_cf"] = "RMSE CF"
mapping["time_sample_obs"] = "Time sample"
mapping["time_log_prob"] = "Time eval"
mapping["time_forward"] = "Time training"
# mapping['param_count'] = 'Params'

col_order = []
col_order.append("kl_forward")
col_order.append("rmse_ate")
col_order.append("rmse_cf")
col_order.append("time_forward")
col_order.append("time_log_prob")
col_order.append("time_sample_obs")
# col_order.append('param_count')

df_latex = df_latex[col_order].rename(mapping, axis=1)


# %%

body = ""

model_dict = {"CausalMAF": r"\ours", "MAF": r"CAREFL$^\dag$", "VACA": "VACA"}

row_list = []
for dataset in df_latex.index.get_level_values("Dataset").unique():
    body = ""
    df_dataset = df_latex[df_latex.index.get_level_values("Dataset") == dataset]
    num_models = len(df_dataset)
    for i, model in enumerate(df_dataset.index.get_level_values("Model")):
        row = df_dataset.loc[(dataset, model)]
        row_str = ""
        if i == 0:
            row_str = "\n" + r"\multirow{" + str(num_models) + r"}{*}{" + dataset + "}"

        row_str += (
            "& " + model_dict[model] + " & " + " & ".join(row.values) + r" \\" + "\n"
        )
        body += row_str
    row_list.append(body)

body = "\n\\midrule\n".join(row_list)
print(body)

# %%

import pandas as pd


# def print_latex_table(df, table_size='normal'):
#     # Define the table size
#     if table_size == 'tiny':
#         table_size_str = r"\tiny"
#     elif table_size == 'small':
#         table_size_str = r"\small"
#     else:
#         table_size_str = ""
#
#     # Define the header and footer of the table
#     header = r"\begin{table}[ht]" + "\n" + \
#              r"\centering" + "\n" + \
#              r"\caption{Model performance on different datasets}" + "\n" + \
#              r"\label{tab:model_performance}" + "\n" + \
#              r"{" + table_size_str + r" \begin{tabular}{l  c | " + "c " * len(df.columns) + "} \\\\" + "\n" + \
#              r"\toprule" + "\n" + \
#              r" \multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Model}} & \multicolumn{" + \
#              str(len(df.columns)) + r"}{c}{\textbf{Metrics}} \\\\" + "\n" + \
#              r" & & & " + " & ".join(["\\textbf{" + metric + "}" for metric in df.columns]) + " \\\\" + "\n" + \
#              r"\midrule" + "\n"
#     footer = r"\bottomrule" + "\n" + \
#              r"\end{tabular}" + r"}"   "\n" + \
#              r"\end{table}"
#
#     # Define the body of the table
#     body = ""
#     for dataset in  df_latex.index.get_level_values('Dataset').unique():
#         df_dataset = df_latex[df_latex.index.get_level_values('Dataset') == dataset]
#         num_models = len(df_dataset)
#         for i, model in enumerate(df_dataset.index.get_level_values('Model')):
#             row = df_dataset.loc[(dataset, model)].iloc[0]
#             row_str = ''
#             if i == 0:
#                 row_str = r"\multirow{" + str(num_models) + r"}{*}{" + dataset + "}"
#
#             row_str += "& " + model + " & ".join(df_dataset.loc[(model, dataset)].values) + r" \\" + "\n"
#
#     for (dataset, model), row in df.iterrows():
#         row = ''
#         num_models = len(df[df.index.get_level_values('Dataset') == dataset])
#         if last_dataset != dataset:
#             last_dataset = dataset
#             row += r"\multirow{" + str(num_models) + r"}{*}{\rotatebox[origin=c]{90}{" + dataset + "}}"
#         else:
#
#         if dataset in df.index:
#             body += r"\multirow{" + str(len(df.loc[(sem, dataset)])) + r"}{*}{\rotatebox[origin=c]{90}{" + sem + "}}" + \
#                     r" & \multirow{" + str(len(df.loc[(sem, dataset)])) + r"}{*}{\rotatebox[origin=c]{90}{" + dataset + "}}"
#             for i, model in enumerate(df.loc[(sem, dataset)].index):
#                 if i != 0:
#                     body += "\n"
#                 if i == 0:
#                     body += " & " + model + " & " + " & ".join([str(x) for x in df.loc[(sem, dataset, model)]]) + " \\\\"
#                 else:
#                     body += "& & " + model + " & " + " & ".join([str(x) for x in df.loc[(sem, dataset, model)]]) + " \\\\"
#             body += r"\midrule" + "\n"
#         body += "\n"
#
#     # Print the complete LaTeX table
#     print(header + body + footer)


# print_latex_table(df_latex, 'tiny')


# %%
