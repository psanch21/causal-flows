import os

import pytest

from causal_nf.preparators.scm import SCMPreparator


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
            if preparator.type == "torch":
                assert len(batch) == 2
                x_norm = scaler.transform(batch[0])
            elif preparator.type == "pyg":
                x = batch.x.reshape(batch.num_graphs, -1)
                x_norm = scaler.transform(x)


@pytest.mark.parametrize(
    "name", ["chain", "chain-4", "chain-5", "collider", "fork", "triangle"]
)
@pytest.mark.parametrize("sem_name", ["linear"])
@pytest.mark.parametrize("num_samples", [128, 1024])
@pytest.mark.parametrize("scale", [None, "default"])
@pytest.mark.parametrize("splits", [[0.8, 0.1, 0.1], [0.8, 0.2]])
@pytest.mark.parametrize("base_version", [1, 2, 3])
@pytest.mark.parametrize("type", ["pyg", "torch"])
def test_scm_preparator_linear(
    name, sem_name, num_samples, scale, splits, base_version, type
):
    root_dir = os.path.join("..", "Data")

    preparator = SCMPreparator(
        name=name,
        num_samples=num_samples,
        sem_name=sem_name,
        base_version=base_version,
        splits=splits,
        shuffle_train=True,
        single_split=False,
        k_fold=-1,
        root=root_dir,
        loss="default",
        type=type,
        scale=scale,
    )

    assert_preparator(preparator)


@pytest.mark.parametrize(
    "name", ["chain", "triangle", "diamond", "fork", "simpson", "large-backdoor"]
)
@pytest.mark.parametrize("sem_name", ["non-linear"])
@pytest.mark.parametrize("num_samples", [128, 1024])
@pytest.mark.parametrize("scale", [None, "default"])
@pytest.mark.parametrize("splits", [[0.8, 0.1, 0.1], [0.8, 0.2]])
@pytest.mark.parametrize("type", ["pyg", "torch"])
def test_scm_preparator_non_linear(name, sem_name, num_samples, scale, splits, type):
    root_dir = os.path.join("..", "Data")

    preparator = SCMPreparator(
        name=name,
        num_samples=num_samples,
        sem_name=sem_name,
        base_version=1,
        splits=splits,
        shuffle_train=True,
        single_split=False,
        k_fold=-1,
        root=root_dir,
        loss="default",
        type=type,
        scale=scale,
    )

    assert_preparator(preparator)


@pytest.mark.parametrize("name", ["simpson"])
@pytest.mark.parametrize("sem_name", ["sym-prod"])
@pytest.mark.parametrize("num_samples", [128, 1024])
@pytest.mark.parametrize("scale", [None, "default"])
@pytest.mark.parametrize("splits", [[0.8, 0.1, 0.1], [0.8, 0.2]])
@pytest.mark.parametrize("base_version", [1, 2, 3])
@pytest.mark.parametrize("type", ["pyg", "torch"])
def test_scm_preparator_sym_prod(
    name, sem_name, num_samples, scale, splits, base_version, type
):
    root_dir = os.path.join("..", "Data")

    preparator = SCMPreparator(
        name=name,
        num_samples=num_samples,
        sem_name=sem_name,
        base_version=base_version,
        splits=splits,
        shuffle_train=True,
        single_split=False,
        k_fold=-1,
        root=root_dir,
        loss="default",
        type=type,
        scale=scale,
    )

    assert_preparator(preparator)
