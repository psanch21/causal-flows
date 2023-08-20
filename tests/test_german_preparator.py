import os

import matplotlib.pyplot as plt
import pytest
from tueplots import bundles

from causal_nf.preparators.german_preparator import GermanPreparator

plt.rcParams.update(bundles.icml2022())


def assert_preparator(preparator):
    preparator.prepare_data()
    scaler = preparator.get_scaler(fit=True)
    loaders = preparator.get_dataloaders(batch_size=16)

    diameter = preparator.diameter()
    assert isinstance(diameter, int)
    longest_path_length = preparator.longest_path_length()
    assert isinstance(longest_path_length, int)
    for loader in loaders:
        for batch in iter(loader):
            assert len(batch) == 2
            x_norm = scaler.transform(batch[0])


@pytest.mark.parametrize("scale", [None, "default"])
@pytest.mark.parametrize("add_noise", [True, False])
@pytest.mark.parametrize("k_fold", [1, 2, 3])
def test_german_preparator(scale, add_noise, k_fold):
    root_dir = os.path.join("..", "Data")

    preparator = GermanPreparator(
        splits=[0.8, 0.1, 0.1],
        shuffle_train=True,
        single_split=False,
        k_fold=k_fold,
        root=root_dir,
        loss="default",
        scale=scale,
        add_noise=add_noise,
    )

    assert_preparator(preparator)


@pytest.mark.parametrize("add_noise", [True, False])
def _test_german_plot(add_noise):
    root_dir = os.path.join("..", "Data")

    preparator = GermanPreparator(
        splits=[0.8, 0.1, 0.1],
        shuffle_train=True,
        single_split=False,
        k_fold=1,
        root=root_dir,
        loss="default",
        scale=None,
        add_noise=add_noise,
    )

    assert_preparator(preparator)

    plt.close("all")
    plt.rcParams.update(bundles.icml2022())

    for split in ["train"]:

        preparator.plot_data(
            split=split,
            num_samples=256,
            folder=os.path.join("images", "samples"),
            filename=f"german_{split}_noise={add_noise}_samples.png",
            save=False,
        )
