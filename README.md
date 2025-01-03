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

### **Experiments**

Our experiments focus on training and evaluation. After installing the package locally, you can run the `train` and `evaluate` scripts, both configurable via the Hydra interface.

---

#### **Training**
To start training, specify the dataset directory (in `ImageFolder` format):

```bash
train train.dataset_dir=/path/to/dataset/train
```

Override parameters as needed, e.g.,:

```bash
train teacher_temp=0.08
```

For a full list of configurable parameters, run:

```bash
train --help
```

The trained model is saved in `model_dir`. Logs are stored in `outputs/`, and metrics are tracked with MLflow in the `runs/` directory. Visualize metrics with:

```bash
mlflow ui --backend-store-uri path/to/runs
```

---

#### **Evaluation**
To evaluate a model, specify the dataset and the model weights.
For the dataset two subdirectories `dataset_train` (=train) and `dataset_val` (=val) are expected:

```bash
evaluate dataset_dir=/path/to/dataset backbone.weights=/path/to/weights
```

This script performs linear probing and k-NN evaluation by default. For customization options, run:

```bash
evaluate --help
```

It also supports basic supervised learning.

---

**Note:** Logs are stored in `outputs/`, and metrics are tracked with MLflow. Both scripts allow flexible configuration using Hydra.


### Development

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
