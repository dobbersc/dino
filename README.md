# DINO

Implementation of the DINO Framework from "Emerging Properties in Self-Supervised Vision Transformers"

## Prerequisites

This project requires Python 3.10 or later.
To install the `dino` Python package, clone this GitHub repository and simply install the package using pip:

```bash
git clone https://github.com/dobbersc/dino
pip install ./dino
```
This installation also contains the experiment scripts and evaluation results.

**Cloning repo on the cluster:**
```bash
# ensure the identity is added on your local machine
ssh-add ~/.ssh/id_rsa 

# ssh agent port forwarding / Alternatively set ForwardAgent yes in your host config
ssh -A hydra 

git clone git@github.com:dobbersc/dino.git
```


**Installing package:**

If you only are interested in the Python package directly (without the experiments), install the `dino`
package directly from GitHub using pip:

```bash
pip install git+https://github.com/dobbersc/dino@master
```

#### Experiments

Our experiments cover the training, fine-tuning and evaluation entry points. The following example commands illustrate their usage: 

- Training: `dino cmd=train train.dataset_dir=/path/to/dataset/train/`
- Fine-Tuning: `dino cmd=finetune dataset@finetune.dataset=imagenet finetune.dataset.data_dir=/path/to/dataset/train/`
- Evaluation: `dino cmd=evaluate evaluate=knn dataset@evaluate.dataset=imagenet evaluate.dataset.data_dir=/path/to/dataset/test`

*Note: We provide some SLURM scripts in the `script` directory.*

#### Development

For development, install the package, including the development dependencies:

```bash
git clone https://github.com/dobbersc/dino
pip install -e ./dino[dev]
```

#### Jupyter-Notebooks
To keep version control clean, only `.py` files are tracked, and `.ipynb` notebooks are ignored. To recreate a notebook from a `.py` file with [jupytext](https://jupytext.readthedocs.io/en/latest/using-cli.html):

1. **Convert `.py` to `.ipynb`:**
    ```bash
    jupytext --to ipynb your_notebook.py
    ```
2. **Set up Syncing:**
    ```bash
    jupytext --set-formats ipynb,py:percent your_notebook.ipynb
    jupytext --sync src/**/*.ipynb # this has to be manually triggered
    ```
3. **Setup vscode task (optional):** 
    ```json
    // .vscode/tasks.json
    {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "jupytext: Sync all .ipynb files",
                "type": "shell",
                "command": ".venv/bin/jupytext --sync src/**/*.ipynb",
                "group": "build"
            }
        ]
    }
    ```

4. **Jupytext vscode extension**
Install Jupytext for Notebooks (congyiwu) and open .py files as notebook.

## Apptainer Containers

We provide Apptainer definition files in the `containers` directory.
For a standard installation, build the container by executing the following from the root of the repository:

```bash
srun --partition=cpu-2h --pty bash
apptainer build containers/dino.sif containers/dino.def
```

#### Development

For an installation in a development environment, build the container by executing the following commands from the root of the repository:

```bash
srun --partition=cpu-2h --pty bash
apptainer build containers/dino-dev.sif containers/dino-dev.def
export APPTAINER_BINDPATH="${PWD}:/dino"
```

*Note that the bind path environment variable must be set for each development session when using the container.*
