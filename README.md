# Causal normalizing flows: from theory to practice

## Description

Welcome to the repository to reproduce the results of the article
["Causal Normalizing Flows: From Theory to Practice"](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8402301e7f06bdc97a31bfaa653dc32-Abstract-Conference.html).
This repository contains all the source code used in our experiments. We appreciate your interest in our work and hope
you find it valuable.

The implementation of the normalizing flows found in the `zuko` folder is a slightly modified version of
[Zuko v0.2.0](https://github.com/probabilists/zuko/releases/tag/0.2.0) (MIT license), which we provide here for the
ease of reproducing the experiments of the article. If you plan to use this code for future work, note that this is
an outdated version of the software, and we encourage you to use the latest version instead.

## Citation

If you use the code or findings from this repository and/or article in your work, please cite the following:

```
@inproceedings{javaloy2023causal,
    title={Causal normalizing flows: from theory to practice},
    author={Adri{\'a}n Javaloy and Pablo Sanchez Martin and Isabel Valera},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=QIFoCI7ca1}
}
```


## Installation

To get started, please follow these installation instructions:

Create a new conda environment by running the following command in your terminal:

```bash
conda create --name causal_nf python=3.9.12 --no-default-packages
 ```

Activate the conda environment using the following command:

```bash
conda activate causal_nf
 ```

Install the additional project requirements using pip and the requirements.txt file:

```bash
pip install torch==1.13.1 torchvision==0.14.1

pip install torch_geometric==2.3.1
pip install torch-scatter==2.1.1

pip install -r requirements.txt
```

Once the installation is complete, you can proceed to run the tests to ensure everything is set up correctly. Run the
following commands in your terminal:

```bash
pytest tests/test_causal_transform.py
pytest tests/test_flows.py
pytest tests/test_german_preparator.py
pytest tests/test_gnn.py
pytest tests/test_scm.py
pytest tests/test_scm_dataset.py
pytest tests/test_scm_preparator.py
```

## Causal Graphs of the Structural Causal Models

To generate the plots of the causal graphs for the Structural Causal Models (SCMs), you can use the following command:

```bash
pytest tests/test_scm_plots.py -k test_scm_plot_graph
```

## Usage

To train the Causal Normalizing Flow (Causal NF) using the provided code, run the main script with the desired
configuration. For example, to train the Causal NF using a specific configuration file:

   ```bash
   python main.py --config_file causal_nf/configs/causal_nf.yaml --wandb_mode disabled --project CAUSAL_NF
   ```

Adjust the `config_file`, `wandb_mode`, and other options as needed for your experimentation.

Feel free to modify the configuration, experiment with different parameters, and tailor the process to your
requirements. Please note that these instructions assume a generic setup; make sure to adjust paths, filenames, and
commands according to your actual codebase.

## Ablation of the direction of the flow

To create the experiments for the ablation study, please execute the following commands:

```bash
python generate_jobs.py --grid_file grids/causal_nf/ablation_u_x/base.yaml --format shell --jobs_per_file 20000 --batch_size 4
python generate_jobs.py --grid_file grids/causal_nf/ablation_x_u/base.yaml --format shell --jobs_per_file 20000 --batch_size 4
```

These commands will generate the necessary experiment configurations based on the specified YAML files. The resulting
experiments will be ready for execution.

Next, let's generate the comparison figure for abductive vs. generative approaches. Execute the following command:

```bash
python scripts/create_figure_ablation_direction.py
```

This command will generate a figure that visually compares the performance of abductive and generative approaches.

Now, let's move on to generating the figure that compares the two best abductive approaches with the two best generative
approaches. Run the following command:

```bash
python scripts/create_figure_ablation_best.py
```

Lastly, to generate the figure that compares the training time for the different approaches, execute the following
command:

```bash
python scripts/create_figure_ablation_time.py
```

## Ablation of the Base Distribution

Create the experiments by executing the following command in your terminal:

```bash
python generate_jobs.py --grid_file grids/causal_nf/base/base.yaml --format shell --jobs_per_file 20000 --batch_size 4
```

This command will generate the necessary experiment configurations based on the provided YAML file. The experiments will
be created and ready for execution.

Next, let's generate the figure that illustrates the ablation of the base distribution. Run the following command:

```bash
python scripts/create_figure_ablation_base.py
```

## Ablation of Flow Architecture

Create the experiments by executing the following command in your terminal:

```bash
python generate_jobs.py --grid_file grids/causal_nf/flow/base.yaml --format shell --jobs_per_file 20000 --batch_size 1
```

This command will generate the necessary experiment configurations based on the provided YAML file. The experiments will
be created and ready for execution.

Next, let's generate the figure that illustrates the ablation of the flow architecture. Run the following command:

```bash
python scripts/create_figure_ablation_flow.py
```

## Model comparison

Create the experiments for the proposed Causal NF by executing the following command:

```bash
python generate_jobs.py --grid_file grids/causal_nf/comparison_x_u/base.yaml --format shell --jobs_per_file 20000 --batch_size 4

```

This command will generate the necessary experiment configurations for the Causal NF based on the provided YAML file.
The experiments will be created and ready for execution.

Similarly, create the experiments for CAREFL by running the following command:

```bash
python generate_jobs.py --grid_file grids/causal_nf/comparison_carefl/base.yaml --format shell --jobs_per_file 20000 --batch_size 4
```

Next, create the experiments for VACA with the following command:

```bash
python generate_jobs.py --grid_file grids/causal_nf/comparison_vaca/base.yaml --format shell --jobs_per_file 20000 --batch_size 2
```

Now, let's generate the comparison table by executing the following commands:

```bash
python scripts/create_comparison_flows.py
python scripts/create_comparison_vaca.py
python scripts/create_comparison_table.py
```

Finally, to generate the pairplot figure, run the following commands:

For the Simpson non-linear pairplot:

```bash
python main.py --config_file causal_nf/configs/causal_nf.yaml --wandb_mode disabled --project CAUSAL_NF --opts model.plot False 

python main.py --load_model output_testing/9068c700f8c011edbd6bacde48001122 --opts model.plot True
```

For the Chain-5 linear pairplot:

```bash
python main.py --config_file causal_nf/configs/causal_nf.yaml --wandb_mode disabled --project CAUSAL_NF --opts model.plot False dataset.name chain-5 dataset.sem_name linear

python main.py --load_model output_testing/8fa58e5efa3011edaac7a683e775b962 --opts model.plot True
```

These commands will generate the pairplot figures for the respective datasets.

## Use case: fairness auditing and classification

To perform fairness auditing and classification using the German dataset, please follow the instructions below:

Download the processed data from [here](https://zenodo.org/records/10785677).

Run the following commands to train the models on 5 different splits of the German dataset:

```bash
python main.py --config_file causal_nf/configs/german_cf.yaml --wandb_mode disabled --project CAUSAL_NF --opts model.plot False dataset.k_fold 1
python main.py --config_file causal_nf/configs/german_cf.yaml --wandb_mode disabled --project CAUSAL_NF --opts model.plot False dataset.k_fold 2
python main.py --config_file causal_nf/configs/german_cf.yaml --wandb_mode disabled --project CAUSAL_NF --opts model.plot False dataset.k_fold 3
python main.py --config_file causal_nf/configs/german_cf.yaml --wandb_mode disabled --project CAUSAL_NF --opts model.plot False dataset.k_fold 4
python main.py --config_file causal_nf/configs/german_cf.yaml --wandb_mode disabled --project CAUSAL_NF --opts model.plot False dataset.k_fold 5 
```

After training the models, run the following command to generate the results for the fairness auditing and
classification:

```bash
python counterfactual_fairness_batch.py --folder output_cf
```

Additionally, you can generate the table with the summary of the unfairness and performance by executing the following
command:

```bash
python counterfactual_fairness_table.py
```

Finally, to generate the pair plot, execute the following command:

```bash
python main.py --load_model output_cf/14b02ca8f93211ed9c2facde48001122 --opts model.plot True
```

## Contact Information

If you have any questions, feedback, or inquiries about the code or the research, feel free to reach
out to the authors: [psanchez@tue.mpg.de](mailto:psanchez@tue.mpg.de)
and [ajavaloy@cs.uni-saarland.de](mailto:ajavaloy@cs.uni-saarland.de)

For issues related to the repository or code, you can also create a GitHub issue or pull request.

We appreciate your interest in our research and code! Your feedback and collaboration are valuable to us.
