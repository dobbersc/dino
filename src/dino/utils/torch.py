import torch
from torch import nn

from dino import config


def detect_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_module_device(module: nn.Module) -> torch.device:
    """Returns the module's device.

    This function requires all submodules to be on the same device.

    Returns:
        The module's device.
    """
    return next(module.parameters()).device


def save_model(model, model_name):
    # check if the model directory exists
    if not config.MODEL_DIR.exists():
        config.MODEL_DIR.mkdir()
    model_path = config.MODEL_DIR / model_name
    # check if the model file exists
    if model_path.exists():
        # create a new file name
        model_name = model_name.split(".")[0] + "_new.pth"
    torch.save(model.state_dict(), model_path)


def load_model(model_name):
    model_path = config.MODEL_DIR / model_name
    return torch.load(model_path)
