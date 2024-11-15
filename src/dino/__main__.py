# my_package/__main__.py


from dataclasses import dataclass
from enum import Enum

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from dino.finetuning import FinetuningConfig, run_finetuning


class Command(Enum):
    train = "train"
    evaluate = "evaluate"
    finetune = "finetune"


@dataclass
class DinoConfig:
    cmd: Command = MISSING
    finetune: FinetuningConfig = MISSING
    verbose: bool = False
    log_dir: str = MISSING


_cs = ConfigStore.instance()
_cs.store(
    name="base_dino",
    node=DinoConfig,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def entry_point(cfg: DinoConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Call the appropriate function based on the command

    match cfg.cmd:
        case Command.train:
            # train(cfg)
            pass
        case Command.evaluate:
            # evaluate(cfg)
            pass
        case Command.finetune:
            run_finetuning(cfg.finetune)


if __name__ == "__main__":
    entry_point()
