import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 300


def get_metrics(log_path: Path) -> tuple[list[float], list[float], list[float], list[float]]:
    log_text: str = log_path.read_text(encoding="utf-8")

    loss: list[float] = [
        float(match.group("loss")) for match in re.finditer(r"Loss:\s(?P<loss>[-+]?\d*\.\d+)", log_text)
    ]
    student_entropies: list[float] = [
        float(match.group("kl_divergence"))
        for match in re.finditer(r"Teacher\sEntropy:\s(?P<kl_divergence>[-+]?\d*\.\d+)", log_text)
    ]
    teacher_entropies: list[float] = [
        float(match.group("kl_divergence"))
        for match in re.finditer(r"Student\sEntropy:\s(?P<kl_divergence>[-+]?\d*\.\d+)", log_text)
    ]
    kl_divergences: list[float] = [
        float(match.group("kl_divergence"))
        for match in re.finditer(r"KL\sDivergence:\s(?P<kl_divergence>[-+]?\d*\.\d+)", log_text)
    ]

    return loss, student_entropies, teacher_entropies, kl_divergences


def main() -> None:
    imagenette: Path = Path("/vol/tmp/dobbersc/PyCharmProjects/dino/rerun/imagenette2/deit.out")
    imagenet100: Path = Path("/vol/tmp/dobbersc/PyCharmProjects/dino/rerun/imagenet100/deit.out")
    imagenet_kaggle: Path = Path("/vol/tmp/dobbersc/PyCharmProjects/dino/rerun/imagenet-kaggle/deit.out")

    figure, ax = plt.subplots()
    for label, log_path in zip(
        ["Imagenette", "ImageNet100", "ImageNet (Kaggle)"], [imagenette, imagenet100, imagenet_kaggle]
    ):
        loss, _, _, _ = get_metrics(log_path)
        loss = loss[:10]
        ax.plot(np.arange(1, len(loss) + 1), loss, marker="o", alpha=0.7, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.legend()
    figure.tight_layout()
    figure.show()

    Path("plots").mkdir(parents=True, exist_ok=True)
    figure.savefig("plots/loss.pdf")

    figure, ax = plt.subplots()
    for label, log_path in zip(
        ["Imagenette", "ImageNet100", "ImageNet (Kaggle)"], [imagenette, imagenet100, imagenet_kaggle]
    ):
        _, _, _, kl_divergences = get_metrics(log_path)
        kl_divergences = kl_divergences[:10]
        ax.plot(np.arange(1, len(kl_divergences) + 1), kl_divergences, marker="o", alpha=0.7, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")

    ax.legend()
    figure.tight_layout()
    figure.show()

    Path("plots").mkdir(parents=True, exist_ok=True)
    figure.savefig("plots/kl-divergence.pdf")

    figure, axes = plt.subplots(3, sharex=True)
    for ax, label, log_path in zip(
        axes, ["Imagenette", "ImageNet100", "ImageNet (Kaggle)"], [imagenette, imagenet100, imagenet_kaggle]
    ):
        _, student_entropies, teacher_entropies, _ = get_metrics(log_path)
        student_entropies = student_entropies[:10]
        teacher_entropies = teacher_entropies[:10]

        xrange = np.arange(1, len(student_entropies) + 1)
        ax.plot(xrange, student_entropies, alpha=0.7, linestyle="--", marker="o", label="Student")
        ax.plot(xrange, teacher_entropies, alpha=0.7, marker="o", label="Teacher")

        ax.set_ylim(7.5, 9)
        ax.set_title(f"[{label}]", fontsize="medium")

    ax.legend(loc="lower right")

    figure.supxlabel("Epoch")
    figure.supylabel("Entropy")

    figure.tight_layout()
    figure.show()

    Path("plots").mkdir(parents=True, exist_ok=True)
    figure.savefig("plots/entropy.pdf")


if __name__ == "__main__":
    main()
