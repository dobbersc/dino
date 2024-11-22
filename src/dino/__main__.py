import logging
from dataclasses import dataclass
from enum import Enum

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from dino.entry_points.evaluation import EvaluatorConfig, run_evaluation
from dino.entry_points.finetune import FinetuningConfig, run_finetuning
from dino.entry_points.train import TrainingConfig, train

logger: logging.Logger = logging.getLogger(__name__)


class Command(Enum):
    train = "train"
    evaluate = "evaluate"
    finetune = "finetune"


@dataclass
class DinoConfig:
    cmd: Command = MISSING
    finetune: FinetuningConfig = MISSING
    evaluate: EvaluatorConfig = MISSING
    train: TrainingConfig = MISSING
    verbose: bool = False
    log_dir: str = MISSING


_cs = ConfigStore.instance()
_cs.store(
    name="base_dino",
    node=DinoConfig,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def entry_point(cfg: DinoConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    # Call the appropriate function based on the command

    match cfg.cmd:
        case Command.train:
            train(cfg.train)
        case Command.evaluate:
            run_evaluation(cfg.evaluate)
        case Command.finetune:
            run_finetuning(cfg.finetune)


if __name__ == "__main__":
    entry_point()
