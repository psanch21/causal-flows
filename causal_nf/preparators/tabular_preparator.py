import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from tueplots import bundles

import causal_nf.utils.io as causal_io
from causal_nf.preparators.base_preparator import BaseDatasetPreparator

plt.rcParams.update(bundles.icml2022())


def plot_tabular(
    batch=None,
    title_elem_idx=None,
    batch_size=None,
    df=None,
    hue=None,
    columns=None,
    diag_plot="kde",
    title="Tabular data",
):
    if batch is None:
        assert df is not None
    else:
        if not isinstance(batch, list):
            batch = [batch]  # Make it a list

        x = batch[0]
        if isinstance(batch_size, int):
            x = x[:batch_size, :]

        df = pd.DataFrame(x.numpy(), columns=columns)

    if isinstance(title_elem_idx, int):
        title_el = batch[title_elem_idx]

    g = sns.PairGrid(df, diag_sharey=False, hue=hue)
    sns.set(font_scale=1.5)
    sns.set(style="white", rc={"axes.grid": False})

    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    if diag_plot == "kde":
        g.map_diag(sns.kdeplot, lw=2)
    elif diag_plot == "hist":
        g.map_diag(sns.histplot)

    # Change font size of labels and ticks
    label_font_size = 14
    tick_font_size = 12

    for ax in g.axes.flat:
        # Set tick font size
        ax.tick_params(axis="both", labelsize=tick_font_size)

        # Set xlabel font size
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_font_size)

        # Set ylabel font size
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_font_size)

    g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
    g.fig.suptitle(title)
    return g.fig


class TabularPreparator(BaseDatasetPreparator):
    def __init__(self, name, **kwargs):

        super().__init__(name=name, **kwargs)

    @classmethod
    def params(cls, dataset):
        if isinstance(dataset, dict):
            dataset = causal_io.dict_to_cn(dataset)

        my_dict = {}

        my_dict.update(BaseDatasetPreparator.params(dataset))

        return my_dict

    @classmethod
    def loader(cls, dataset):
        my_dict = TabularPreparator.params(dataset)

        return cls(**my_dict)

    @property
    def type_of_data(self):
        return "tabular"

    def _metric_names(self):
        return []

    def _batch_element_list(self):
        raise NotImplementedError

    def feature_names(self, latex=False):
        if latex:
            return [f"$x_{{{i + 1}}}$" for i in range(self.x_dim())]
        else:
            return [f"x_{i + 1}" for i in range(self.x_dim())]

    def _data_loader(self, dataset, batch_size, shuffle, num_workers=0):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )

    def create_df(self, x_list, name_list=None):
        df_list = []
        for i, x in enumerate(x_list):
            df = pd.DataFrame(x.numpy(), columns=self.feature_names(True))
            if isinstance(name_list, list):
                name_i = name_list[i]
                df["mode"] = name_i
            df_list.append(df)

        df = pd.concat(df_list)
        return df

    def post_process(self, x):
        return x

    def _plot_data(
        self,
        batch=None,
        title_elem_idx=None,
        batch_size=None,
        df=None,
        hue=None,
        title=None,
        diag_plot="kde",
        **kwargs,
    ):

        if title is None:
            title = f"Dataset: {self.name}"

        return plot_tabular(
            batch=batch,
            title_elem_idx=title_elem_idx,
            batch_size=batch_size,
            df=df,
            hue=hue,
            columns=self.feature_names(True),
            diag_plot=diag_plot,
            title=title,
        )

    def _split_dataset(self, dataset_raw):
        raise NotImplementedError

    def get_dataset_train(self):
        return self.datasets[0]

    def get_features_train(self):
        loader = self.get_dataloader_train(batch_size=self.num_samples())
        batch = next(iter(loader))
        return batch[0]

    def get_scaler_info(self):
        if self.scale in ["default", "min0_max1"]:
            return [("min0_max1", None)]
        elif self.scale in ["minn1_max1"]:
            return [("minn1_max1", None)]
        elif self.scale in ["std"]:
            return [("std", None)]
        else:
            raise NotImplementedError

    def num_samples(self):
        return len(self.datasets[0])
