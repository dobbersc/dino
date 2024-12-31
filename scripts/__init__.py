from pathlib import Path

import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 300

MLFLOW_DIR: Path = Path("runs")
EXPORT_PATH: Path | None = Path("plots")
