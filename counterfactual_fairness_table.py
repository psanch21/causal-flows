# %%
import glob

import numpy as np
import pandas as pd

folder = "output_cf"

files = glob.glob(f"{folder}/**/counterfactual_fairness.pickle", recursive=True)
df_list = []

for file in files:
    df_i = pd.read_pickle(file)
    df_list.append(df_i)

df = pd.concat(df_list).copy()

df = df.rename(columns={"cf_unfairness": "unfairness"})
# Define your desired order
clf_order = ["logistic", "svc"]
data_order = ["full", "unaware", "fair", "fair_z"]


metrics = ["f1", "accuracy", "unfairness"]



r = "".join(
    [
        "r",
    ]
    * len(data_order)
)


header = f"{{\\tiny \\begin{{tabular}}{{l | {r} | {r} }} \\\\ \n"
header += "\\toprule \n"
#
col_clf = " & ".join(
    [
        f"\\multicolumn{{{len(data_order)}}}{{c}}{{\\textbf{{ {c} }} }}"
        for c in clf_order
    ]
)
col_clf_lef = "\\multirow{2}{*}{\\textbf{Metric}}"
header += f" {col_clf_lef} & {col_clf} \\\\ \n"
col_headers = " & ".join([f"$\\mathrm{{ {c} }}$" for c in data_order])
header += f" & {col_headers}  & {col_headers} \\\\ \n"
header += "\\midrule \n"

print(header)
body = ""
for metric in metrics:
    my_str = f" {metric} "
    for clf in clf_order:
        for data in data_order:
            df_clf = df[df.clf == clf]
            df_data = df_clf[df_clf.data == data]
            df_metric = df_data[metric] * 100
            mean = df_metric.mean()
            std = df_metric.std()
            my_str += f" & {mean:.2f}$_{{\pm{std:.2f}}}$"

    body += f"{my_str} \\\\\n"

print(body)
footer = "\\bottomrule \n"
footer += "\\end{tabular}} \n"
print(footer)
