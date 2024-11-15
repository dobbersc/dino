from pathlib import Path

import torch
from torch import nn


def detect_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_module_device(module: nn.Module) -> torch.device:
    """Returns the module's device.

    This function requires all submodules to be on the same device.

    Returns:
        The module's device.
    """
    return next(module.parameters()).device


def save_model(model, model_path: str | Path):
    # ensure that the directory exists
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # check if it has a .pth extension
    if model_path.suffix != ".pth":
        model_path = model_path.with_suffix(".pth")

    # check if the model file exists
    if model_path.exists():
        # create a new file name
        model_path = model_path.with_name(model_path.stem + "_new" + model_path.suffix)

    torch.save(model.state_dict(), model_path)


def load_model(model_path: str | Path):
    return torch.load(model_path)
