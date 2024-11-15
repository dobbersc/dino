import shutil
from pathlib import Path

import pytest


@pytest.fixture
def tmp_model_dir(tmp_path: Path):
    """Creates a temporary model directory for testing and cleans it up after the test."""
    # Create a temporary directory within the test's tmp_path
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Yield the directory path to the test function
    yield model_dir

    # Cleanup: Remove the directory and its contents after the test
    shutil.rmtree(model_dir, ignore_errors=True)
