import os

import pytest

from causal_nf.config import get_config_default_file
from causal_nf.modules import module_dict
from causal_nf.preparators.scm import SCMPreparator


@pytest.mark.parametrize("layer_name", ["gcn", "gat", "gin", "pna"])
@pytest.mark.parametrize("use_edge_attr", [False, True])
def test_gnn_default_config(layer_name, use_edge_attr):
    cfg = get_config_default_file(as_dict=False)
    root_dir = os.path.join("..", "Data")

    preparator = SCMPreparator(
        name="chain",
        num_samples=246,
        sem_name="linear",
        base_version=1,
        splits=[0.8, 0.1, 0.1],
        shuffle_train=True,
        single_split=False,
        k_fold=-1,
        root=root_dir,
        loss="default",
        type="pyg",
        use_edge_attr=use_edge_attr,
        scale=None,
    )

    preparator.prepare_data()

    GNN = module_dict[layer_name]
    kwargs = GNN.kwargs(cfg, preparator)

    gnn = GNN(**kwargs)

    loader = preparator.get_dataloader_train(batch_size=8)
    batch = next(iter(loader))

    logits = gnn(batch)
    assert logits.shape[0] == batch.x.shape[0]
