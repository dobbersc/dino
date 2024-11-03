from pathlib import Path

import torch

ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "models"

IMAGENET_TINY_DIR = DATA_DIR / "tiny-imagenet-200" / "train"
IMAGENET_TINY_WORDS = DATA_DIR / "tiny-imagenet-200" / "words.txt"


def save_model(model, model_name):
    # check if the model directory exists
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir()
    model_path = MODEL_DIR / model_name
    # check if the model file exists
    if model_path.exists():
        # create a new file name
        model_name = model_name.split(".")[0] + "_new.pth"
    torch.save(model.state_dict(), model_path)


def load_model(model_name):
    model_path = MODEL_DIR / model_name
    return torch.load(model_path)
