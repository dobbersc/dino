from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "models"

IMAGENET_TINY_DIR = DATA_DIR / "tiny-imagenet-200" / "train"
IMAGENET_TINY_WORDS = DATA_DIR / "tiny-imagenet-200" / "words.txt"

IMAGENET_DIR = "/input-data"