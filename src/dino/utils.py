import os
import torch
from dino import config

def save_model(model, model_name):
    # check if the model directory exists
    if not config.MODEL_DIR.exists():
        MODEL_DIR.mkdir()
    model_path = config.MODEL_DIR / model_name
    # check if the model file exists
    if model_path.exists():
        # create a new file name
        model_name = model_name.split(".")[0] + "_new.pth"
    torch.save(model.state_dict(), model_path)


def load_model(model_name):
    model_path = config.MODEL_DIR / model_name
    return torch.load(model_path)




def list_directory_contents(directory_path):
    """
    List all files and directories inside a specified directory.

    Args:
        directory_path (str): Path to the directory.
    """
    
    directory_path = str(directory_path)
    for root, dirs, files in os.walk(directory_path):
        # Calculate the depth of the current directory to format the output
        depth = root.replace(directory_path, "").count(os.sep)
        indent = " " * 4 * depth
        print(f"{indent}{os.path.basename(root)}/")

        # List directories
        sub_indent = " " * 4 * (depth + 1)
        for d in dirs:
            print(f"{sub_indent}{d}/")

        # List files
        for f in files:
            print(f"{sub_indent}{f}")
