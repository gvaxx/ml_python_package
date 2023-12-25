import hydra
from omegaconf import DictConfig

import mlops_project.train_rfr as train_rfr


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    train_rfr.main(cfg)


if __name__ == "__main__":
    main()
