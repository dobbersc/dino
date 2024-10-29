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
If you only are interested in the Python package directly (without the experiments), install the `dino`
package directly from GitHub using pip:

```bash
pip install git+https://github.com/dobbersc/dino@master
```

#### Experiments

...

#### Development

For development, install the package, including the development dependencies:

```bash
git clone https://github.com/dobbersc/dino
pip install -e ./dino[dev]
```

## Apptainer Containers

We provide Apptainer definition files in the `containers` directory.
For a standard installation, build the container by executing the following from the root of the repository:

```bash
apptainer build containers/dino.sif containers/dino.def
```

#### Development

For an installation in a development environment, build the container by executing the following commands from the root of the repository:

```bash
apptainer build containers/dino-dev.sif containers/dino-dev.def
export APPTAINER_BINDPATH="${PWD}:/dino"
```

*Note that the bind path environment variable must be set for each development session when using the container.*
