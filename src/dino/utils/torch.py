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
