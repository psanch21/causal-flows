import sys

sys.path.append("../")
sys.path.append("./")

import os

import scripts.helpers as script_help

import pandas as pd

pd.set_option("display.max_columns", None)


import causal_nf.utils.list_op as list_op

import matplotlib.pyplot as plt

from tueplots import bundles

plt.rcParams.update(bundles.icml2022())
# plt.rcParams.update(figsizes.icml2022_full())

import warnings

warnings.filterwarnings("ignore")
import re

import seaborn as sns
import matplotlib.pyplot as plt

root = "output_causal_nf"
folder = os.path.join("results", "images")

keep_cols = []
keep_cols.append("dataset__name")
keep_cols.append("dataset__sem_name")
keep_cols.append("dataset__num_samples")
keep_cols.append("dataset__base_version")

keep_cols.append("model__name")
keep_cols.append("model__layer_name")
keep_cols.append("model__dim_inner")
keep_cols.append("model__adjacency")
keep_cols.append("model__base_to_data")
keep_cols.append("model__base_distr")

keep_cols.append("train__regularize")

# %% Load dataframes
df_all = []
for exp_folder in ["ablation_u_x", "ablation_x_u"]:
    df = script_help.load_df(root, [exp_folder], keep_cols, freq=10)
    df_all.append(df.training)

df = pd.concat(df_all, axis=0)

# %%

pattern = re.compile(".*time.*", flags=re.IGNORECASE)

cols = [
    "dataset__name",
    "dataset__sem_name",
    "model__layer_name",
    "model__adjacency",
    "model__num_layers",
    "model__base_to_data",
    "train__regularize",
    "split",
]

cols = list_op.list_intersection(cols, df.columns)
regex = "|".join(cols) + "|" + pattern.pattern
print(regex)
df_ = df.copy()
df_ = df_.filter(regex=regex)
df_ = df_[df_.model__num_layers == 5]
df_ = script_help.update_names(df_)

df_ = df_.rename(columns={"train__regularize": "Regularize"})

# %% Print the datasets and models under comparison

print(f"Datasets")
for d in df_.Dataset.unique():
    print(f"\t{d}")
print(f"Models")
for d in df_.Model.unique():
    print(f"\t{d}")

# %% Prepare data for plotting
cols = ["Dataset", "Model", "Direction", "time_forward"]
df_plot = df_.copy()
df_plot = df_plot[df_plot.split == "train"][cols]

mapping = {key: int(key[-6]) for key in df_plot.Dataset.unique()}
df_plot["Dataset"] = df_plot["Dataset"].map(mapping)

y = r"Training time $(\mu s)$"
x = r"d"
df_plot = df_plot.rename(columns={"time_forward": y, "Dataset": x})

# %%

plt.rcParams.update(bundles.icml2022())
from tueplots import fontsizes

double_ = {}
for key, value in fontsizes.icml2022().items():
    double_[key] = 1.8 * value
plt.rcParams.update(double_)

# %%

filename = f"ablation_time.pdf"
fig, ax = plt.subplots()
for (direction, model_name), df_g in df_plot.groupby(["Direction", "Model"]):
    print(f"Direction: {direction} Model: {model_name}")
    color = script_help.select_color(model_name)
    linestyle = script_help.select_style(direction)
    marker = script_help.select_marker(model_name)
    x_ticks = sorted(df_g[x].unique())
    df_g[x] = df_g[x].map({x_ticks[i]: i for i in range(len(x_ticks))})

    sns.lineplot(
        data=df_g,
        x=x,
        y=y,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markerfacecolor=color,
        markeredgecolor="white" if marker == "*" else color,
        markersize=15 if marker == "*" else None,
        markeredgewidth=1.0 if marker == "*" else None,
    )
    ax.grid(True)
    ax.set_xticks(list(range(len(x_ticks))))
    ax.set_xticklabels(x_ticks)

ax.set_ylim((0.0, None))
ax.set_xlabel(r"Size of graph $(d)$")
path = os.path.join(folder, filename)
plt.tight_layout()
fig.savefig(path)
# plt.show()

# %% Create legend
# For the legend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Create a figure and axis
fig, ax = plt.subplots()

lines = []

models = ["MAF", "MAF*", "CausalMAF", "CausalMAF*"]
m_dict = {}
m_dict["MAF"] = r"Ordering"
m_dict["MAF*"] = r"Ordering$^\star$"
m_dict["CausalMAF"] = r"Graph"
m_dict["CausalMAF*"] = r"Graph$^\star$"

# Create dummy lines for legend without colors
dummy_line_1 = Line2D([], [], linestyle="", color="k", label="Model:")
lines.append(dummy_line_1)
for m in models:
    marker = script_help.select_marker(m)
    color = script_help.select_color(m.replace("*", ""))
    dummy_line_1 = Line2D(
        [],
        [],
        lw=3,
        linestyle="-",
        marker=marker,
        color=color,
        label=m_dict[m],
        markerfacecolor=color,
        markeredgecolor="white" if marker == "*" else color,
        markersize=15 if marker == "*" else 8,
        markeredgewidth=1.0 if marker == "*" else None,
    )
    lines.append(dummy_line_1)

# Create the legend without colors
ax.legend(handles=lines, ncols=1, frameon=False)

# Remove unnecessary plot elements
ax.axis("off")
plt.tight_layout()
# Save the legend as an image file
plt.savefig(os.path.join(folder, "ablation_time_legend_color.pdf"))

# Show the legend
# plt.show()


# Create a figure and axis
fig, ax = plt.subplots()
lines = []

# Create dummy lines for legend without colors
dummy_line_1 = Line2D([], [], linestyle="", color="k", label="Direction:")
lines.append(dummy_line_1)
dummy_line_1 = Line2D(
    [],
    [],
    lw=1.3,
    linestyle="-",
    color="k",
    label=r"$\mathbf{x} \rightarrow \mathbf{u}$",
)
lines.append(dummy_line_1)
dummy_line_1 = Line2D(
    [],
    [],
    lw=1.3,
    linestyle=":",
    color="k",
    label=r"$\mathbf{u} \rightarrow \mathbf{x}$",
)
lines.append(dummy_line_1)

# Create the legend without colors
ax.legend(handles=lines, ncols=1, frameon=False)

# Remove unnecessary plot elements
ax.axis("off")
plt.tight_layout()
# Save the legend as an image file
plt.savefig(os.path.join(folder, "ablation_time_legend_style.pdf"))

# Show the legend
# plt.show()

plt.close("all")
