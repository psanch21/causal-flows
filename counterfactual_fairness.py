# %%
import causal_nf.utils.wandb_local as wandb_local
import causal_nf.config as causal_nf_config
from causal_nf.config import cfg
import causal_nf.utils.training as causal_nf_train
from causal_nf.preparators.german_preparator import GermanPreparator
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", type=str, default=None)

args = parser.parse_args()

load_model = args.load_model
c_file, c_d_file = wandb_local.get_configs_from_folder(load_model)

config = causal_nf_config.build_config(
    config_file=c_file, args_list=None, config_default_file=c_d_file
)

cfg.model.base_distr = "normal"
config["model__base_distr"] = "normal"

causal_nf_config.assert_cfg_and_config(cfg, config)

# %%
assert cfg.dataset.name in ["german"]

preparator = GermanPreparator.loader(cfg.dataset)
preparator.prepare_data()

ckpt_name_list = glob.glob(os.path.join(load_model, f"*ckpt"))

for ckpt_name_i in ckpt_name_list:
    print(ckpt_name_i)

# %%

ckpt_file = ckpt_name_list[0]
model_lightning = causal_nf_train.load_model(
    cfg=cfg, preparator=preparator, ckpt_file=ckpt_file
)

model = model_lightning.model
model.eval()

# %%

scaler = preparator.get_scaler()

obs_dict = model.sample((10000,))

x_obs_norm = obs_dict["x_obs"]
x_obs = scaler.inverse_transform(x_obs_norm, inplace=False)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import numpy as np

seed = 57

datasets = {}
x_train, y_train = preparator.datasets[0].data(one_hot=True, scaler=scaler)
x_val, y_val = preparator.datasets[1].data(one_hot=True, scaler=scaler)
x_test, y_test = preparator.datasets[2].data(one_hot=True, scaler=scaler)

datasets_cf = {}
for i, dataset_i in enumerate(preparator.datasets):
    loader = preparator._data_loader(
        dataset_i, batch_size=len(dataset_i), shuffle=False, num_workers=0
    )

    x, y = next((iter(loader)))
    print("---")
    output_cf = {}
    output_cf_i = model.compute_counterfactual(
        x, index=0, value=0.0, scaler=preparator.scaler_transform, return_dict=True
    )

    x_cf_0 = preparator.post_process(output_cf_i["x_cf"])
    z_cf_0 = output_cf_i["z_cf"]

    output_cf_i = model.compute_counterfactual(
        x, index=0, value=1.0, scaler=preparator.scaler_transform, return_dict=True
    )

    x_cf = preparator.post_process(output_cf_i["x_cf"])
    z_cf = output_cf_i["z_cf"]

    z_factual_i = output_cf_i["z_factual"]

    x = preparator.post_process(x)
    x_cf[x[:, 0] == 1, :] = x_cf_0[x[:, 0] == 1, :]
    z_cf[x[:, 0] == 1, :] = z_cf_0[x[:, 0] == 1, :]
    x_cf_norm, _ = preparator.datasets[0].data(one_hot=True, scaler=scaler, x=x_cf)
    output_cf[f"x_cf"] = x_cf_norm
    output_cf[f"z_factual"] = z_factual_i
    output_cf[f"z_cf"] = z_cf
    datasets_cf[preparator.split_names[i]] = output_cf

# %%

datasets["full"] = {}
datasets["full"]["x_train"] = x_train
datasets["full"]["y_train"] = y_train
datasets["full"]["x_val"] = x_val
datasets["full"]["y_val"] = y_val
datasets["full"]["x_test"] = x_test
datasets["full"]["y_test"] = y_test
datasets["full"]["x_cf"] = datasets_cf["test"]["x_cf"]

mask_unnamed = list(range(x_train.shape[1]))[1:]

datasets["unaware"] = {}
datasets["unaware"]["x_train"] = x_train[..., mask_unnamed]
datasets["unaware"]["y_train"] = y_train
datasets["unaware"]["x_val"] = x_val[..., mask_unnamed]
datasets["unaware"]["y_val"] = y_val
datasets["unaware"]["x_test"] = x_test[..., mask_unnamed]
datasets["unaware"]["y_test"] = y_test
datasets["unaware"]["x_cf"] = datasets_cf["test"]["x_cf"][..., mask_unnamed]

mask_fair = [1]

datasets["fair"] = {}
datasets["fair"]["x_train"] = x_train[..., mask_fair]
datasets["fair"]["y_train"] = y_train
datasets["fair"]["x_val"] = x_val[..., mask_fair]
datasets["fair"]["y_val"] = y_val
datasets["fair"]["x_test"] = x_test[..., mask_fair]
datasets["fair"]["y_test"] = y_test
datasets["fair"]["x_cf"] = datasets_cf["test"]["x_cf"][..., mask_fair]

z_train = datasets_cf["train"]["z_factual"]
z_val = datasets_cf["val"]["z_factual"]
z_test = datasets_cf["test"]["z_factual"]
mask_fair = list(range(z_train.shape[1]))[1:]

datasets["fair_z"] = {}
datasets["fair_z"]["x_train"] = z_train[..., mask_fair]
datasets["fair_z"]["y_train"] = y_train
datasets["fair_z"]["x_val"] = z_val[..., mask_fair]
datasets["fair_z"]["y_val"] = y_val
datasets["fair_z"]["x_test"] = z_test[..., mask_fair]
datasets["fair_z"]["y_test"] = y_test
datasets["fair_z"]["x_cf"] = datasets_cf["test"]["z_cf"][..., mask_fair]

classifiers = {}

param_grid = [
    {"C": [0.2, 0.5, 1.0], "penalty": ["l2"], "class_weight": ["balanced"]},
    {"C": [1.0], "penalty": [None], "class_weight": ["balanced"]},
]
classifiers["logistic"] = (LogisticRegression, param_grid)

param_grid = [
    {
        "C": [0.2, 0.5, 1],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "class_weight": ["balanced"],
        "probability": [True],
    },
]
classifiers["svc"] = (SVC, param_grid)

# %%

df = pd.DataFrame(
    columns=["clf", "data", "seed", "f1", "accuracy", "best_params_", "cf_unfairness"]
)

for data_name, data in datasets.items():
    print(f"Data: {data_name}")
    for clf_name, (Clf, param_grid) in classifiers.items():
        print(f"Clf: {clf_name}")
        for seed in [252]:
            x_train = data["x_train"]
            y_train = data["y_train"]
            x_val = data["x_val"]
            y_val = data["y_val"]
            x_test = data["x_test"]
            y_test = data["y_test"]
            x_test_cf = data["x_cf"]
            x_tr_val = np.concatenate((x_train, x_val))
            y_tr_val = np.concatenate((y_train, y_val))
            split_index = [-1] * len(x_train) + [0] * len(x_val)

            clf_base = Clf(random_state=seed)

            pds = PredefinedSplit(test_fold=split_index)
            clf = GridSearchCV(clf_base, param_grid=param_grid)

            search = clf.fit(x_tr_val, y_tr_val)

            y_pred = clf.predict(x_test)

            score_f1 = f1_score(y_test, y_pred)
            score_acc = accuracy_score(y_test, y_pred)

            y_proba = clf.predict_proba(x_test)
            y_proba_cf = clf.predict_proba(x_test_cf)

            diff = y_proba[:, 1] - y_proba_cf[:, 1]
            cf_unfairness = np.abs(diff).mean()
            new_row = pd.DataFrame(
                {
                    "clf": [clf_name],
                    "data": [data_name],
                    "seed": [seed],
                    "f1": [score_f1],
                    "accuracy": [score_acc],
                    "best_params_": [str(search.best_params_)],
                    "cf_unfairness": [cf_unfairness],
                }
            )
            print(f"\tSeed: {seed}")
            print(f"\t{new_row.to_dict()}")
            df = pd.concat([df, new_row], ignore_index=True)

# %%

filename = os.path.join(load_model, "counterfactual_fairness.pickle")

df.to_pickle(filename)
# %%
df = pd.read_pickle(filename)
for clf_name in df.clf.unique():
    print(clf_name)
    df_i = df[df.clf == clf_name]

    df_pivot = pd.pivot_table(
        df_i,
        values=["f1", "accuracy", "cf_unfairness"],
        index=None,
        aggfunc=np.mean,
        columns="data",
    )

    df_pivot = df_pivot.reindex(
        columns=[
            "full",
            "unaware",
            "fair",
            "fair_z",
        ]
    )
    df_pivot = df_pivot.iloc[[0, 2, 1], :]
    print((df_pivot * 100).round(2))
# %%