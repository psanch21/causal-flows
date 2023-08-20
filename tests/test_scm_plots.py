import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from tueplots import bundles

from causal_nf.sem_equations import sem_dict

plt.rcParams.update(bundles.icml2022())
from causal_nf.preparators.scm import SCMPreparator
import os


def plot(sem_fn, name, extra=""):
    adj_matrix = sem_fn.adjacency().T

    # Create a graph object from the adjacency matrix
    G = nx.from_numpy_matrix(np.array(adj_matrix), create_using=nx.DiGraph)

    # Customize the appearance of the graph
    # pos = nx.spring_layout(G, seed=42, k=0.5)  # Use a fixed seed for reproducibility
    pos = nx.spectral_layout(G)  # Use a fixed seed for reproducibility
    # pos = nx.planar_layout(G, scale=-0.5, center=None)  # Use a fixed seed for reproducibility
    # pos = nx.kamada_kawai_layout(G)  # Use a fixed seed for reproducibility
    node_colors = "lightgray"
    node_border_colors = "black"
    edge_colors = "black"

    if len(adj_matrix) > 4:
        node_size = 500
        font_size = 12
    else:
        node_size = 1000
        font_size = 16
    node_labels = {i: f"$x_{i + 1}$" for i in range(len(adj_matrix))}

    # Plot the graph
    fig, ax = plt.subplots(figsize=bundles.icml2022()["figure.figsize"])

    nx.draw(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size,
        edge_color=edge_colors,
        with_labels=False,
        arrows=True,
        linewidths=1.0,
        edgecolors=node_border_colors,
        ax=ax,
    )

    # Draw node labels with black text
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=font_size,
        font_weight="bold",
        font_color="black",
    )

    # Customize the appearance of the plot
    ax.axis("off")
    ax.margins(0.2)  # Add padding to prevent cutting off elements
    plt.tight_layout()
    plt.savefig(f"images/{name}_graph{extra}.png", dpi=300, bbox_inches="tight")
    plt.close("all")


@pytest.mark.parametrize(
    "name",
    [
        "chain",
        "chain-4",
        "chain-5",
        "collider",
        "fork",
        "large-backdoor",
        "simpson",
        "triangle",
    ],
)
def test_scm_plot_graph(name):
    sem_name = "non-linear"

    if name == "simpson":
        sem_fn = sem_dict[name](sem_name="sym-prod")
        plot(sem_fn, name, extra="_sym-prod")
    elif name in ["chain-5", "chain-4", "collider"]:
        sem_name = "linear"

    sem_fn = sem_dict[name](sem_name=sem_name)
    plot(sem_fn, name)


@pytest.mark.parametrize(
    "name",
    [
        "chain",
        "chain-4",
        "chain-5",
        "collider",
        "diamond",
        "fork",
        "large-backdoor",
        "simpson",
        "triangle",
    ],
)
def test_scm_plot_samples(name):

    if name in ["large-backdoor"]:
        base_version_list = [1]
    else:
        base_version_list = [1, 2, 3]

    sem_name_list = []
    if name in ["chain", "chain-4", "chain-5", "collider", "fork", "triangle"]:
        sem_name_list.append("linear")
    if name in ["chain", "diamond", "fork", "large-backdoor", "simpson", "triangle"]:
        sem_name_list.append("non-linear")
    if name == "simpson":
        sem_name_list.append("sym-prod")
    for sem_name in sem_name_list:
        root_dir = os.path.join("..", "Data")

        for base_version in base_version_list:

            preparator = SCMPreparator(
                name=name,
                num_samples=256,
                sem_name=sem_name,
                base_version=base_version,
                splits=[0.9, 0.1],
                shuffle_train=True,
                single_split=False,
                k_fold=-1,
                root=root_dir,
                loss="default",
                scale=None,
            )

            preparator.prepare_data()

            plt.close("all")
            plt.rcParams.update(bundles.icml2022())

            preparator.plot_data(
                split="train",
                num_samples=256,
                folder=os.path.join("images", "samples"),
                filename=f"{name}_{sem_name}_{base_version}_samples.png",
                save=False,
            )
